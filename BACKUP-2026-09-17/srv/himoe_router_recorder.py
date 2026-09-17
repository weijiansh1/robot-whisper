"""Capture HiMoE-VLA MoE routing state via forward hooks on the gate submodules.

Verified against upstream HiMoE-VLA (src/moevla/models/{modeling_moe,himoe}.py):

  - gemma_expert has 18 layers; layers 0/1/16/17 are ASMoE (3 experts, top-1,
    routed on the 24-dim data_mask), layers 2-5/12-15 are HBMoE (32 experts,
    top-4, routed on the 1024-dim hidden state), 6-11 are dense.  -> 12 gates.
  - MoEGate.forward returns (topk_idx, topk_weight, aux_loss).
  - aux_loss is None outside training.
  - The gates are invoked as ``self.gate(...)`` so submodule hooks fire even
    though moevla.py calls ``paligemma_with_expert.forward(...)`` directly.

Two things this module does differently from the naive version:

  1. combine_weight is NOT stored.  It is exactly derivable:
       HB (top_k=4, norm_topk_prob=True):  combine = raw / raw.sum(-1)
       AS (top_k=1, no normalisation):     combine = raw
     See ``combine_weight_from_raw``.

  2. Nothing is moved to host memory inside the hook.  Per-gate tensors are
     buffered on device and transferred once per control step, which turns
     ~120 syncs/step into ~5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

GATE_CLASSES = {
    "MoEGate_load_bal": "HB",
    "MoEGate_mutual_info": "AS",
}

_LAYER_RE = re.compile(r"layers\.(\d+)\.mlp\.gate$")


@dataclass
class GateInfo:
    name: str
    layer_idx: int
    kind: str
    module: nn.Module
    n_experts: int
    top_k: int


def discover_gates(core: nn.Module) -> list[GateInfo]:
    """Find every HiMoE router in ``core``, ordered by execution order."""
    found: list[GateInfo] = []
    for name, module in core.named_modules():
        kind = GATE_CLASSES.get(module.__class__.__name__)
        if kind is None:
            continue
        match = _LAYER_RE.search(name)
        if match is None:
            raise RuntimeError(
                f"gate {name!r} does not match the expected "
                f"'layers.<i>.mlp.gate' layout; the model structure changed"
            )
        found.append(
            GateInfo(
                name=name,
                layer_idx=int(match.group(1)),
                kind=kind,
                module=module,
                n_experts=module.n_routed_experts,
                top_k=module.top_k,
            )
        )
    found.sort(key=lambda g: g.layer_idx)
    return found


def unwrap(model: nn.Module) -> nn.Module:
    """Peel DDP / DeepSpeed / Accelerate / compile wrappers off ``model``."""
    seen = 0
    while seen < 8:
        for attr in ("module", "_orig_mod", "model"):
            inner = getattr(model, attr, None)
            if isinstance(inner, nn.Module) and inner is not model:
                if discover_gates(inner):
                    model = inner
                    break
        else:
            break
        seen += 1
    return model


def _router_logits(gate: nn.Module, gate_input: torch.Tensor, kind: str) -> torch.Tensor:
    """Reproduce the gate's own logits exactly.

    MoEGate_mutual_info casts its input to bfloat16 (modeling_moe.py:128) while
    gate.weight stays float32 (gemma_expert torch_dtype is "float32"), so this
    only type-checks inside the autocast block that policy.py:71 sets up.  The
    guard below keeps the hook usable if the caller forgot the autocast or cast
    the whole model to bf16 instead.
    """
    x = gate_input.to(torch.bfloat16) if kind == "AS" else gate_input
    w = gate.weight
    if x.dtype != w.dtype and not torch.is_autocast_enabled():
        x = x.to(w.dtype)
    return F.linear(x.reshape(-1, x.shape[-1]), w, None)


def combine_weight_from_raw(raw: torch.Tensor | Any, top_k: int) -> Any:
    """Reconstruct the weights the model actually used to mix expert outputs."""
    if top_k > 1:  # norm_topk_prob is True for both configs
        return raw / (raw.sum(axis=-1, keepdims=True) + 1e-20)
    return raw


@dataclass
class ControlStepRecord:
    """One ``policy.infer`` call: all routers, all denoising steps."""

    episode_id: int
    control_step: int
    n_denoise: int
    # [B, L_hb, D, S, K]
    hb_expert_ids: Any = None
    hb_selected_prob: Any = None
    hb_entropy: Any = None
    hb_router_probs: Any = None  # [B, L_hb, D, S, E] only if requested
    # AS collapses to [B, L_as] / [B, L_as, 3] when the dedup check holds
    as_expert_ids: Any = None
    as_probs: Any = None
    as_collapsed: bool = True
    # router inputs, only when store_hidden: [B, L, D, S, H]
    hb_hidden: Any = None
    as_hidden: Any = None
    hb_layers: list[int] = field(default_factory=list)
    as_layers: list[int] = field(default_factory=list)


class HiMoERouteRecorder:
    """Buffer routing state on device, emit one record per control step."""

    def __init__(
        self,
        core: nn.Module,
        store_full_probs: bool = False,
        expect_gates: int = 12,
        store_hidden: bool = False,
        verify_steps: int = 4,
    ) -> None:
        self.core = unwrap(core)
        self.store_full_probs = store_full_probs
        self.store_hidden = store_hidden
        #: Re-deriving the router's decision from ``args[0]`` is what proves the
        #: hook grabbed the router's real input.  Checked on the first
        #: ``verify_steps`` control steps of every run rather than always: it costs
        #: a topk per gate per denoising step, and a hook that is wrong is wrong
        #: from the first call, not intermittently.
        self.verify_steps = verify_steps
        self.verify_failures: list[str] = []
        self.verified_calls = 0
        self.gates = discover_gates(self.core)
        if expect_gates and len(self.gates) != expect_gates:
            raise RuntimeError(
                f"expected {expect_gates} HiMoE routers, found {len(self.gates)}; "
                f"you probably passed the policy wrapper rather than the MoEVLA model"
            )
        self.hb_layers = [g.layer_idx for g in self.gates if g.kind == "HB"]
        self.as_layers = [g.layer_idx for g in self.gates if g.kind == "AS"]
        self._first_layer = self.gates[0].layer_idx
        self._handles: list[Any] = []
        self._buf: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._denoise = -1
        self.episode_id = -1
        self.control_step = -1
        self.enabled = False

    # -- lifecycle ---------------------------------------------------------
    def attach(self) -> "HiMoERouteRecorder":
        for gate in self.gates:
            self._handles.append(
                gate.module.register_forward_hook(self._make_hook(gate))
            )
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buf.clear()

    def begin_control_step(self, episode_id: int, control_step: int) -> None:
        self.episode_id = episode_id
        self.control_step = control_step
        self._denoise = -1
        self._buf = {g.layer_idx: [] for g in self.gates}
        self.enabled = True

    # -- hook --------------------------------------------------------------
    def _make_hook(self, gate: GateInfo):
        def hook(module, args, output):
            if not self.enabled:
                return None
            if gate.layer_idx == self._first_layer:
                self._denoise += 1

            gate_input = args[0]
            topk_idx, topk_weight, _aux = output
            bsz, seq_len, _ = gate_input.shape

            with torch.no_grad():
                logits = _router_logits(module, gate_input, gate.kind)
                probs = logits.softmax(dim=-1)
                raw = probs.gather(-1, topk_idx)

                # Hook correctness: the experts we would pick from our own
                # recomputed probabilities must be the experts the gate actually
                # returned.  If args[0] were the wrong tensor -- the block input
                # instead of the router input, say -- the probabilities would
                # still look like a plausible distribution and every downstream
                # number would be quietly wrong.  Compare as sets: the gate's
                # top-k order is its own business.
                if self.control_step < self.verify_steps:
                    self.verified_calls += 1
                    mine = probs.topk(gate.top_k, dim=-1).indices
                    same = (mine.sort(-1).values == topk_idx.sort(-1).values).all()
                    if not bool(same):
                        n_bad = int((mine.sort(-1).values
                                     != topk_idx.sort(-1).values).any(-1).sum())
                        self.verify_failures.append(
                            "%s (layer %d, %s): %d/%d token(s) disagree"
                            % (gate.name, gate.layer_idx, gate.kind,
                               n_bad, mine.shape[0])
                        )

                entry = {
                    "idx": topk_idx.reshape(bsz, seq_len, gate.top_k),
                    "raw": raw.reshape(bsz, seq_len, gate.top_k),
                }
                if self.store_hidden:
                    entry["hidden"] = gate_input.reshape(bsz, seq_len, -1)
                if gate.kind == "HB":
                    p32 = probs.float().clamp_min(1e-12)
                    entry["entropy"] = (
                        -(p32 * p32.log()).sum(-1).reshape(bsz, seq_len)
                    )
                if gate.kind == "AS" or self.store_full_probs:
                    entry["probs"] = probs.reshape(bsz, seq_len, gate.n_experts)
            self._buf[gate.layer_idx].append(entry)
            return None

        return hook

    # -- emit --------------------------------------------------------------
    def end_control_step(self) -> ControlStepRecord:
        """Stack the buffer and pull it to host in a handful of transfers."""
        self.enabled = False
        n_denoise = self._denoise + 1
        if n_denoise <= 0:
            raise RuntimeError("no router fired between begin/end_control_step")
        if self.verify_failures:
            raise RuntimeError(
                "router hook does not reproduce the gate's own top-k, so args[0] "
                "is not the tensor the router saw -- every captured probability "
                "would be wrong. Failures: " + "; ".join(self.verify_failures[:4])
            )

        def stack(layers: list[int], key: str) -> torch.Tensor:
            # -> [L, D, B, S, ...] then move B to the front
            per_layer = [torch.stack([e[key] for e in self._buf[l]]) for l in layers]
            out = torch.stack(per_layer)
            return out.permute(2, 0, 1, *range(3, out.ndim))

        rec = ControlStepRecord(
            episode_id=self.episode_id,
            control_step=self.control_step,
            n_denoise=n_denoise,
            hb_layers=list(self.hb_layers),
            as_layers=list(self.as_layers),
        )

        hb_idx = stack(self.hb_layers, "idx")
        rec.hb_expert_ids = hb_idx.to(torch.uint8).cpu().numpy()
        rec.hb_selected_prob = stack(self.hb_layers, "raw").to(torch.float16).cpu().numpy()
        rec.hb_entropy = stack(self.hb_layers, "entropy").to(torch.float16).cpu().numpy()
        if self.store_full_probs:
            rec.hb_router_probs = (
                stack(self.hb_layers, "probs").to(torch.float16).cpu().numpy()
            )

        # AS routing depends only on data_mask, which is loop-invariant across
        # denoising steps (moevla.py:434-442) and shared by all suffix tokens
        # (modeling_moe.py:262).  Verify rather than assume, then collapse.
        as_idx = stack(self.as_layers, "idx")  # [B, L, D, S, 1]
        as_probs = stack(self.as_layers, "probs")  # [B, L, D, S, 3]
        collapsed = bool(
            (as_idx == as_idx[:, :, :1, :1]).all() and torch.allclose(
                as_probs, as_probs[:, :, :1, :1].expand_as(as_probs)
            )
        )
        rec.as_collapsed = collapsed
        if collapsed:
            rec.as_expert_ids = as_idx[:, :, 0, 0, 0].to(torch.uint8).cpu().numpy()
            rec.as_probs = as_probs[:, :, 0, 0, :].to(torch.float16).cpu().numpy()
        else:
            rec.as_expert_ids = as_idx.to(torch.uint8).cpu().numpy()
            rec.as_probs = as_probs.to(torch.float16).cpu().numpy()

        # Hidden states are NOT collapsed even though AS routing is: the AS
        # decision is loop-invariant because it reads data_mask, but the hidden
        # state that reaches the same gate still changes every denoising step.
        if self.store_hidden:
            rec.hb_hidden = stack(self.hb_layers, "hidden").to(torch.float16).cpu().numpy()
            rec.as_hidden = stack(self.as_layers, "hidden").to(torch.float16).cpu().numpy()

        self._buf = {g.layer_idx: [] for g in self.gates}
        return rec
