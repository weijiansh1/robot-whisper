"""Capture true MoE outputs for the first control row of a K-draw pool.

This recorder is intentionally separate from ``himoe_state_recorder.py``.  The
old recorder reduces an entire rollout to norms; this one keeps just enough of
control step zero to compare candidates generated from the same observation:

* AS layers 0/1: the actual selected-expert output on action tokens;
* every HB layer: routed/shared/post-MoE norms and routed/shared cosine;
* every HB layer: post-MoE vectors, held only until a K-candidate group is
  reduced by the writer;
* optional full routed/shared vectors for analyses that need exact directions.

The suffix convention is explicit: token 0 is the state token and tokens
1..n_action_steps are action tokens.  All arrays emitted here have the state
token removed; the writer stores the corresponding token ids as an axis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from himoe_state_recorder import BlockInfo, discover_blocks


@dataclass(frozen=True)
class FirstControlIdentity:
    episode_id: int
    task_id: int
    init_state_id: int
    flow_seed: int
    control_step: int
    flow_noise_sha256: bytes

    def __post_init__(self) -> None:
        if self.control_step != 0:
            raise ValueError("first-control recorder only accepts control_step=0")
        if len(self.flow_noise_sha256) != 32:
            raise ValueError("flow_noise_sha256 must be the 32 raw digest bytes")


@dataclass
class FirstControlOutputRecord:
    identity: FirstControlIdentity
    hb_layers: np.ndarray
    as_layers: np.ndarray
    # Candidate-first arrays.  A is action token id 1..n_action_steps.
    hb_expert_ids: np.ndarray       # [B, Lh, D, A, K]
    hb_combine_weight: np.ndarray   # [B, Lh, D, A, K]
    hb_routed_norm: np.ndarray      # [B, Lh, D, A]
    hb_shared_norm: np.ndarray      # [B, Lh, D, A]
    hb_post_norm: np.ndarray        # [B, Lh, D, A]
    hb_branch_cosine: np.ndarray    # [B, Lh, D, A]
    hb_branch_angle_deg: np.ndarray # [B, Lh, D, A]
    hb_post_output: np.ndarray      # [B, Lh, D, A, H], temporary fp32
    as_expert_ids: np.ndarray       # [B, La, D, A]
    as_output: np.ndarray           # [B, La, D, A, H], fp16
    hb_routed_output: np.ndarray | None = None
    hb_shared_output: np.ndarray | None = None


def flow_noise_digest(noise: Any) -> bytes:
    """Hash the exact float32 C-order flow-noise request sent to the model."""
    value = np.ascontiguousarray(np.asarray(noise, dtype=np.float32))
    return hashlib.sha256(value.tobytes(order="C")).digest()


def seeded_flow_noise_digest(flow_seed: int, shape: tuple[int, ...]) -> bytes:
    """Reproduce rollout_with_routes.py's first noise draw for identity checks."""
    rng = np.random.default_rng(int(flow_seed))
    noise = rng.standard_normal(shape).astype(np.float32)
    return flow_noise_digest(noise)


class FirstControlOutputRecorder:
    """Forward-hook recorder enabled for one policy call at a time."""

    def __init__(
        self,
        core: nn.Module,
        n_action_steps: int,
        expected_denoise: int,
        store_hb_vectors: bool = False,
        expect_blocks: int = 12,
    ) -> None:
        self.core = core
        self.n_action_steps = int(n_action_steps)
        self.expected_denoise = int(expected_denoise)
        self.store_hb_vectors = bool(store_hb_vectors)
        self.blocks = discover_blocks(core)
        if expect_blocks and len(self.blocks) != expect_blocks:
            raise RuntimeError(
                "expected %d MoE blocks, found %d" % (expect_blocks, len(self.blocks))
            )
        self.hb = [b for b in self.blocks if b.kind == "HB"]
        self.as_capture = [b for b in self.blocks if b.kind == "AS" and b.layer_idx in (0, 1)]
        if [b.layer_idx for b in self.as_capture] != [0, 1]:
            raise RuntimeError("expected AS capture layers [0, 1]")
        if any(b.has_shared for b in self.as_capture):
            raise RuntimeError("AS0/1 unexpectedly have shared experts")
        self._first_layer = self.blocks[0].layer_idx
        self._handles: list[Any] = []
        self.enabled = False
        self.identity: FirstControlIdentity | None = None
        self._denoise = -1
        self._batch_size: int | None = None
        self._vector_hidden_size: int | None = None
        self._gate_rows: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._hb_rows: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._as_rows: dict[int, list[torch.Tensor]] = {}
        self._pending_shared: dict[int, torch.Tensor] = {}

    @property
    def hb_layers(self) -> list[int]:
        return [b.layer_idx for b in self.hb]

    @property
    def as_layers(self) -> list[int]:
        return [b.layer_idx for b in self.as_capture]

    def attach(self) -> "FirstControlOutputRecorder":
        for block in self.blocks:
            self._handles.append(
                block.module.gate.register_forward_hook(self._gate_hook(block))
            )
            if block.kind == "HB":
                if not block.has_shared:
                    raise RuntimeError("HB layer %d has no shared expert" % block.layer_idx)
                self._handles.append(
                    block.module.shared_experts.register_forward_hook(
                        self._shared_hook(block)
                    )
                )
                self._handles.append(
                    block.module.register_forward_hook(self._block_hook(block))
                )
            elif block.layer_idx in (0, 1):
                self._handles.append(
                    block.module.register_forward_hook(self._block_hook(block))
                )
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.cancel()

    def begin(self, identity: FirstControlIdentity) -> None:
        if self.enabled:
            raise RuntimeError("a capture is already active")
        self.identity = identity
        self._denoise = -1
        self._batch_size = None
        self._vector_hidden_size = None
        self._gate_rows = {b.layer_idx: [] for b in self.hb + self.as_capture}
        self._hb_rows = {b.layer_idx: [] for b in self.hb}
        self._as_rows = {b.layer_idx: [] for b in self.as_capture}
        self._pending_shared = {}
        self.enabled = True

    def cancel(self) -> None:
        self.enabled = False
        self.identity = None
        self._gate_rows.clear()
        self._hb_rows.clear()
        self._as_rows.clear()
        self._pending_shared.clear()

    def _check_suffix(self, tensor: torch.Tensor, check_vector_width: bool = False) -> None:
        if tensor.ndim != 3:
            raise RuntimeError("expected [batch, suffix, hidden], got %s" % (tensor.shape,))
        if tensor.shape[1] != self.n_action_steps + 1:
            raise RuntimeError(
                "expected one state + %d action tokens, got suffix=%d"
                % (self.n_action_steps, tensor.shape[1])
            )
        batch, _suffix, hidden = tensor.shape
        if self._batch_size is None:
            self._batch_size = int(batch)
        elif batch != self._batch_size:
            raise RuntimeError("batch size changed during capture")
        # AS gates read the 24-dim expanded data_mask, while AS/HB blocks act on
        # the 1024-dim model hidden state.  Gate hooks therefore validate only
        # B/S; block hooks additionally enforce a common vector width.
        if check_vector_width:
            if self._vector_hidden_size is None:
                self._vector_hidden_size = int(hidden)
            elif hidden != self._vector_hidden_size:
                raise RuntimeError("MoE block hidden width changed during capture")

    def _gate_hook(self, block: BlockInfo):
        def hook(_gate, args, output):
            if not self.enabled:
                return None
            if block.layer_idx == self._first_layer:
                self._denoise += 1
            if block.layer_idx not in self._gate_rows:
                return None
            x = args[0]
            self._check_suffix(x, check_vector_width=False)
            topk_idx, combine_weight, _aux = output
            batch, suffix, _hidden = x.shape
            idx = topk_idx.reshape(batch, suffix, block.top_k)[:, 1:].detach()
            weight = combine_weight.reshape(batch, suffix, block.top_k)[:, 1:].detach()
            self._gate_rows[block.layer_idx].append((idx, weight))
            return None

        return hook

    def _shared_hook(self, block: BlockInfo):
        def hook(_module, _args, output):
            if self.enabled:
                self._pending_shared[block.layer_idx] = output.detach()
            return None

        return hook

    def _block_hook(self, block: BlockInfo):
        def hook(_module, args, output):
            if not self.enabled:
                return None
            x = args[0]
            self._check_suffix(x, check_vector_width=True)
            post = output.detach()
            if block.kind == "HB":
                shared = self._pending_shared.pop(block.layer_idx, None)
                if shared is None:
                    raise RuntimeError(
                        "HB layer %d block fired without shared output" % block.layer_idx
                    )
                routed = post - shared.to(post.dtype)
                self._hb_rows[block.layer_idx].append(
                    (routed[:, 1:], shared[:, 1:], post[:, 1:])
                )
            elif block.layer_idx in self._as_rows:
                self._as_rows[block.layer_idx].append(post[:, 1:])
            return None

        return hook

    @staticmethod
    def _stack_vectors(rows, layers: list[int]) -> torch.Tensor:
        # layer, denoise, batch, action, hidden -> batch, layer, denoise, action, hidden
        value = torch.stack([torch.stack(rows[layer], dim=0) for layer in layers], dim=0)
        return value.permute(2, 0, 1, 3, 4).contiguous()

    @staticmethod
    def _stack_routes(rows, layers: list[int], slot: int) -> torch.Tensor:
        # layer, denoise, batch, action, top-k -> batch, layer, denoise, action, top-k
        value = torch.stack(
            [torch.stack([entry[slot] for entry in rows[layer]], dim=0) for layer in layers],
            dim=0,
        )
        return value.permute(2, 0, 1, 3, 4).contiguous()

    def _validate_counts(self) -> None:
        expected = self.expected_denoise
        if self._denoise + 1 != expected:
            raise RuntimeError(
                "captured %d denoise steps, expected %d" % (self._denoise + 1, expected)
            )
        for family in (self._gate_rows, self._hb_rows, self._as_rows):
            wrong = {layer: len(rows) for layer, rows in family.items() if len(rows) != expected}
            if wrong:
                raise RuntimeError("incomplete per-layer capture: %s" % wrong)
        if self._pending_shared:
            raise RuntimeError("unconsumed shared outputs: %s" % sorted(self._pending_shared))

    def end(self) -> FirstControlOutputRecord:
        if not self.enabled or self.identity is None:
            raise RuntimeError("no active capture")
        self.enabled = False
        self._validate_counts()

        hb_layers = self.hb_layers
        as_layers = self.as_layers
        routed = self._stack_vectors(
            {layer: [row[0] for row in self._hb_rows[layer]] for layer in hb_layers},
            hb_layers,
        )
        shared = self._stack_vectors(
            {layer: [row[1] for row in self._hb_rows[layer]] for layer in hb_layers},
            hb_layers,
        )
        post = self._stack_vectors(
            {layer: [row[2] for row in self._hb_rows[layer]] for layer in hb_layers},
            hb_layers,
        )
        as_output = self._stack_vectors(self._as_rows, as_layers)
        hb_idx = self._stack_routes(self._gate_rows, hb_layers, 0)
        hb_weight = self._stack_routes(self._gate_rows, hb_layers, 1)
        as_idx = self._stack_routes(self._gate_rows, as_layers, 0).squeeze(-1)

        with torch.no_grad():
            routed_f = routed.float()
            shared_f = shared.float()
            post_f = post.float()
            branch_cosine = F.cosine_similarity(
                routed_f, shared_f, dim=-1, eps=1e-12
            ).clamp(-1.0, 1.0)
            record = FirstControlOutputRecord(
                identity=self.identity,
                hb_layers=np.asarray(hb_layers, dtype=np.int16),
                as_layers=np.asarray(as_layers, dtype=np.int16),
                hb_expert_ids=hb_idx.to(device="cpu", dtype=torch.uint8).numpy(),
                hb_combine_weight=hb_weight.to(device="cpu", dtype=torch.float16).numpy(),
                hb_routed_norm=routed_f.norm(dim=-1).cpu().numpy(),
                hb_shared_norm=shared_f.norm(dim=-1).cpu().numpy(),
                hb_post_norm=post_f.norm(dim=-1).cpu().numpy(),
                hb_branch_cosine=branch_cosine.cpu().numpy(),
                hb_branch_angle_deg=torch.rad2deg(torch.acos(branch_cosine)).cpu().numpy(),
                # Kept only for one K pool.  Preserve the model tensor exactly
                # under an fp32 upcast so the group reduction is not fp16-approximate.
                hb_post_output=post.to(device="cpu", dtype=torch.float32).numpy(),
                as_expert_ids=as_idx.to(device="cpu", dtype=torch.uint8).numpy(),
                as_output=as_output.to(device="cpu", dtype=torch.float16).numpy(),
                hb_routed_output=(
                    routed.to(device="cpu", dtype=torch.float16).numpy()
                    if self.store_hb_vectors else None
                ),
                hb_shared_output=(
                    shared.to(device="cpu", dtype=torch.float16).numpy()
                    if self.store_hb_vectors else None
                ),
            )

        self.identity = None
        self._gate_rows.clear()
        self._hb_rows.clear()
        self._as_rows.clear()
        self._pending_shared.clear()
        return record
