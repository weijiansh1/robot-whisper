"""Capture HB-MoE amplitudes plus one runtime pre-gate vector probe.

This records an activation, not a router score.  For every HB layer and flow
round it intercepts the return value of ``HBMoE.moe_infer``.  That tensor is the
actual sum of the selected expert outputs after multiplication by the gate's
returned top-k weights.  In the released model those weights have already been
renormalised by ``norm_topk_prob``.

Only action tokens are retained (suffix token zero is the state token), and the
hidden axis is reduced on device before anything is copied to the CPU.  In
addition to branch RMS values, the recorder evaluates each selected expert on
the exact runtime input and records ``RMS(E_e(h))`` before the upstream in-place
gate multiplication.  This deliberately adds expert compute while capturing;
it is a measurement path, not a latency benchmark.  The resulting arrays are::

    [batch, 8 HB layers, 10 flow rounds, n_action_steps]
    [batch, 8 HB layers, 10 flow rounds, n_action_steps, top_k]

For one explicitly selected HB layer (layer 5 in production), denoise round zero
also stores the exact runtime input, shared branch, full router softmax, and all
selected pre-gate expert vectors.  Expert hooks clone those vectors before
upstream ``moe_infer`` mutates its local output with ``mul_``.  Their weighted
fp32 reconstruction must match the actual routed tensor before storage downcast.

The routed RMS is computed *after* the weighted expert vectors have been summed,
so cancellation between experts is part of the measured contribution.  Shared
and total-MoE RMS are captured as controls.  Router probabilities are never used
as a proxy for activation magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from himoe_state_recorder import BlockInfo, discover_blocks


@dataclass
class HBActivationRecord:
    """One policy query, with the state suffix token removed."""

    episode_id: int
    control_step: int
    query_id: int
    candidate_id: int
    n_denoise: int
    hb_layers: np.ndarray
    hb_routed_rms: np.ndarray
    hb_shared_rms: np.ndarray
    hb_total_mlp_rms: np.ndarray
    hb_topk_weight_sum_error: np.ndarray
    hb_selected_expert_id: np.ndarray
    hb_selected_expert_weight: np.ndarray
    hb_selected_expert_raw_rms: np.ndarray
    probe_hb_layer: int
    probe_denoise: int
    hb_probe_input_hidden: np.ndarray
    hb_probe_shared_output: np.ndarray
    hb_probe_selected_expert_raw: np.ndarray
    hb_probe_router_probs: np.ndarray
    hb_probe_routed_reconstruction_max_abs_error: np.ndarray
    intervention_pair_id: int = -1
    intervention_draw_id: int = -1
    intervention_arm: str = "none"
    intervention_drop_policy: str = "none"
    hb_probe_original_routed: np.ndarray | None = None
    hb_probe_executed_routed: np.ndarray | None = None
    hb_probe_intervention_delta: np.ndarray | None = None
    hb_probe_dropped_slot: np.ndarray | None = None
    hb_probe_dropped_expert_id: np.ndarray | None = None
    hb_probe_internal_l2: np.ndarray | None = None
    hb_probe_internal_l1: np.ndarray | None = None
    hb_probe_internal_signed_mean: np.ndarray | None = None
    hb_probe_internal_positive_fraction: np.ndarray | None = None
    hb_probe_internal_linf: np.ndarray | None = None


def _rms_last(x: torch.Tensor) -> torch.Tensor:
    """RMS over hidden width, in fp32 even when the model runs in bf16."""
    return x.float().square().mean(dim=-1).sqrt()


def _hb_router_probs(gate: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Recompute the HB gate softmax from the exact gate input."""
    weight = gate.weight
    value = hidden
    if value.dtype != weight.dtype and not torch.is_autocast_enabled():
        value = value.to(weight.dtype)
    logits = F.linear(value.reshape(-1, value.shape[-1]), weight, None)
    return logits.softmax(dim=-1)


class HBActivationRMSRecorder:
    """Capture true routed/shared HB-MoE amplitudes for each policy query.

    ``attach`` temporarily wraps each HB block's ``moe_infer`` method.  The
    original method still performs all dispatch and combination; the wrapper
    only retains its returned tensor until the enclosing HB block hook fires.
    """

    _METRICS = (
        "routed_rms",
        "shared_rms",
        "total_mlp_rms",
        "topk_weight_sum_error",
    )
    _INTERNAL_METRICS = (
        "internal_l2",
        "internal_l1",
        "internal_signed_mean",
        "internal_positive_fraction",
        "internal_linf",
    )

    def __init__(
        self,
        core: nn.Module,
        n_action_steps: int,
        expected_denoise: int,
        expect_hb_blocks: int = 8,
        probe_layer: int = 5,
        reconstruction_atol: float = 1e-5,
        reconstruction_rtol: float = 1e-5,
        intervention: Any | None = None,
        capture_internal_activation: bool = False,
    ) -> None:
        self.core = core
        self.n_action_steps = int(n_action_steps)
        self.expected_denoise = int(expected_denoise)
        if self.n_action_steps <= 0 or self.expected_denoise <= 0:
            raise ValueError("n_action_steps and expected_denoise must be positive")

        self.hb: list[BlockInfo] = [
            block for block in discover_blocks(core) if block.kind == "HB"
        ]
        if expect_hb_blocks and len(self.hb) != expect_hb_blocks:
            raise RuntimeError(
                "expected %d HB-MoE blocks, found %d" % (expect_hb_blocks, len(self.hb))
            )
        if not self.hb:
            raise RuntimeError("no HB-MoE blocks found")
        missing_shared = [b.layer_idx for b in self.hb if not b.has_shared]
        if missing_shared:
            raise RuntimeError("HB layers lack shared experts: %s" % missing_shared)
        invalid_norm = [
            b.layer_idx
            for b in self.hb
            if b.top_k > 1 and not bool(getattr(b.module.gate, "norm_topk_prob", False))
        ]
        if invalid_norm:
            raise RuntimeError(
                "HB top-k weights are not configured for normalisation at layers %s"
                % invalid_norm
            )

        self.probe_layer = int(probe_layer)
        probe = [block for block in self.hb if block.layer_idx == self.probe_layer]
        if len(probe) != 1:
            raise RuntimeError(
                "expected exactly one HB probe layer %d, found %d"
                % (self.probe_layer, len(probe))
            )
        self._probe_block = probe[0]
        self.probe_denoise = 0
        self.reconstruction_atol = float(reconstruction_atol)
        self.reconstruction_rtol = float(reconstruction_rtol)
        if self.reconstruction_atol < 0.0 or self.reconstruction_rtol < 0.0:
            raise ValueError("reconstruction tolerances must be non-negative")
        gate_weight = getattr(self._probe_block.module.gate, "weight", None)
        if gate_weight is None or gate_weight.ndim != 2:
            raise RuntimeError("HB probe gate must expose a 2-D weight tensor")
        self.probe_hidden_size = int(gate_weight.shape[1])
        self.probe_n_experts = int(self._probe_block.n_experts)
        self.probe_top_k = int(self._probe_block.top_k)
        self.intervention = intervention
        self.capture_internal_activation = bool(capture_internal_activation)
        if intervention is not None and (
            int(intervention.layer) != self.probe_layer
            or int(intervention.denoise) != self.probe_denoise
        ):
            raise ValueError("intervention cell must match the recorder probe cell")
        self.probe_intermediate_size = 0
        if self.capture_internal_activation:
            widths = set()
            for expert in self._probe_block.module.experts:
                down_proj = getattr(expert, "down_proj", None)
                if down_proj is None or not hasattr(down_proj, "in_features"):
                    raise RuntimeError(
                        "internal activation capture requires expert.down_proj"
                    )
                config = getattr(expert, "config", None)
                if config is not None and int(getattr(config, "pretraining_tp", 1)) != 1:
                    raise RuntimeError(
                        "down_proj pre-hook is invalid when pretraining_tp is not one"
                    )
                widths.add(int(down_proj.in_features))
            if len(widths) != 1:
                raise RuntimeError("probe experts do not share one intermediate width")
            self.probe_intermediate_size = widths.pop()

        self._handles: list[Any] = []
        self._original_moe_infer: dict[int, Callable[..., torch.Tensor]] = {}
        self._wrapper: dict[int, Callable[..., torch.Tensor]] = {}
        self._had_instance_method: dict[int, bool] = {}
        self._instance_method: dict[int, Any] = {}
        self._rows: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._pending_routed: dict[int, torch.Tensor] = {}
        self._pending_weight_error: dict[int, torch.Tensor] = {}
        self._pending_shared: dict[int, torch.Tensor] = {}
        self._pending_expert_id: dict[int, torch.Tensor] = {}
        self._pending_expert_weight: dict[int, torch.Tensor] = {}
        self._pending_expert_raw_rms: dict[int, torch.Tensor] = {}
        self._pending_probe: dict[int, dict[str, torch.Tensor]] = {}
        self._active_probe_dispatch: dict[str, Any] | None = None
        self.episode_id = -1
        self.control_step = -1
        self.query_id = -1
        self.candidate_id = -1
        self.intervention_pair_id = -1
        self.intervention_draw_id = -1
        self.intervention_arm = "none"
        self.intervention_drop_policy = "none"
        self.enabled = False
        self.attached = False

    @property
    def hb_layers(self) -> list[int]:
        return [block.layer_idx for block in self.hb]

    @property
    def top_k(self) -> int:
        values = {block.top_k for block in self.hb}
        if len(values) != 1:
            raise RuntimeError("HB blocks do not share one top-k: %s" % sorted(values))
        return values.pop()

    @property
    def n_routed_experts(self) -> int:
        values = {block.n_experts for block in self.hb}
        if len(values) != 1:
            raise RuntimeError(
                "HB blocks do not share one expert count: %s" % sorted(values)
            )
        return values.pop()

    def _is_probe_forward(self, block: BlockInfo) -> bool:
        return (
            self.enabled
            and block.layer_idx == self.probe_layer
            and len(self._rows[block.layer_idx]) == self.probe_denoise
        )

    def _probe_gate_hook(self, block: BlockInfo):
        def hook(gate, args, output):
            if not self._is_probe_forward(block):
                return None
            if block.layer_idx in self._pending_probe:
                raise RuntimeError("HB vector probe gate fired twice before dispatch")
            hidden = args[0]
            if hidden.ndim != 3:
                raise RuntimeError(
                    "HB vector probe gate input must be [batch,suffix,hidden]"
                )
            topk_idx, topk_weight, _aux = output
            with torch.no_grad():
                probs = _hb_router_probs(gate, hidden)
                expected = (hidden.shape[0] * hidden.shape[1], block.n_experts)
                if probs.shape != expected:
                    raise RuntimeError(
                        "HB vector probe router shape %s, expected %s"
                        % (tuple(probs.shape), expected)
                    )
                recomputed = probs.topk(block.top_k, dim=-1).indices.sort(-1).values
                actual = topk_idx.reshape(-1, block.top_k).sort(-1).values
                if not bool((recomputed == actual).all()):
                    raise RuntimeError(
                        "HB vector probe full softmax does not reproduce top-k IDs"
                    )
                selected_prob = probs.gather(-1, topk_idx.reshape(-1, block.top_k))
                combine = selected_prob / (
                    selected_prob.sum(dim=-1, keepdim=True) + 1e-20
                )
                actual_weight = topk_weight.reshape(-1, block.top_k).float()
                if not torch.allclose(
                    combine.float(), actual_weight, rtol=1e-5, atol=1e-6
                ):
                    raise RuntimeError(
                        "HB vector probe full softmax does not reproduce top-k weights"
                    )
                self._pending_probe[block.layer_idx] = {
                    "router_probs": probs.reshape(
                        hidden.shape[0], hidden.shape[1], block.n_experts
                    ).detach()
                }
            return None

        return hook

    def _start_probe_dispatch(
        self,
        block: BlockInfo,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
    ) -> None:
        if self._active_probe_dispatch is not None:
            raise RuntimeError("nested HB vector probe dispatch")
        top_k = block.top_k
        sorted_position = flat_expert_indices.argsort()
        sorted_expert = flat_expert_indices[sorted_position]
        token = sorted_position // top_k
        slot = sorted_position % top_k
        dispatch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for expert_id in range(block.n_experts):
            selected = sorted_expert == expert_id
            if bool(selected.any()):
                dispatch[expert_id] = (token[selected], slot[selected])
        self._active_probe_dispatch = {
            "layer": block.layer_idx,
            "raw": torch.full(
                (x.shape[0], top_k, x.shape[1]),
                float("nan"),
                dtype=torch.float32,
                device=x.device,
            ),
            "dispatch": dispatch,
            "seen": set(),
        }
        if self.capture_internal_activation:
            for name in self._INTERNAL_METRICS:
                self._active_probe_dispatch[name] = torch.full(
                    (x.shape[0], top_k),
                    float("nan"),
                    dtype=torch.float32,
                    device=x.device,
                )
            self._active_probe_dispatch["internal_seen"] = set()

    def _probe_down_proj_pre_hook(self, block: BlockInfo, expert_id: int):
        def hook(_down_proj, args):
            context = self._active_probe_dispatch
            if context is None or context["layer"] != block.layer_idx:
                return None
            expected = context["dispatch"].get(expert_id)
            if expected is None:
                raise RuntimeError(
                    "HB internal probe saw unselected expert %d" % expert_id
                )
            if expert_id in context["internal_seen"]:
                raise RuntimeError(
                    "HB internal probe expert %d fired twice" % expert_id
                )
            if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("HB expert down_proj must receive one tensor")
            value = args[0]
            token, slot = expected
            if value.shape != (token.numel(), self.probe_intermediate_size):
                raise RuntimeError(
                    "HB internal activation shape %s is incompatible with expert %d"
                    % (tuple(value.shape), expert_id)
                )
            reduced = value.detach().float()
            context["internal_l2"][token, slot] = reduced.square().sum(-1).sqrt()
            context["internal_l1"][token, slot] = reduced.abs().sum(-1)
            context["internal_signed_mean"][token, slot] = reduced.mean(-1)
            context["internal_positive_fraction"][token, slot] = (
                reduced.gt(0).float().mean(-1)
            )
            context["internal_linf"][token, slot] = reduced.abs().amax(-1)
            context["internal_seen"].add(expert_id)
            return None

        return hook

    def _probe_expert_hook(self, block: BlockInfo, expert_id: int):
        def hook(_expert, _args, output):
            context = self._active_probe_dispatch
            if context is None or context["layer"] != block.layer_idx:
                return None
            expected = context["dispatch"].get(expert_id)
            if expected is None:
                raise RuntimeError(
                    "HB vector probe saw unselected expert %d" % expert_id
                )
            if expert_id in context["seen"]:
                raise RuntimeError("HB vector probe expert %d fired twice" % expert_id)
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise RuntimeError("HB vector probe expert output must be 2-D")
            token, slot = expected
            if output.shape != (token.numel(), context["raw"].shape[-1]):
                raise RuntimeError(
                    "HB vector probe expert %d output shape %s is incompatible"
                    % (expert_id, tuple(output.shape))
                )
            # `moe_infer` may call `.float()` and then mutate the same storage
            # with `mul_`; copy now while this is still the pre-gate value.
            value = output.detach().to(dtype=torch.float32, copy=True)
            context["raw"][token, slot] = value
            context["seen"].add(expert_id)
            return None

        return hook

    def attach(self) -> "HBActivationRMSRecorder":
        if self.attached:
            raise RuntimeError("activation recorder is already attached")
        for block in self.hb:
            layer = block.layer_idx
            module = block.module
            original = module.moe_infer
            self._original_moe_infer[layer] = original
            self._had_instance_method[layer] = "moe_infer" in module.__dict__
            if self._had_instance_method[layer]:
                self._instance_method[layer] = module.__dict__["moe_infer"]

            def traced_moe_infer(
                x: torch.Tensor,
                flat_expert_indices: torch.Tensor,
                flat_expert_weights: torch.Tensor,
                *,
                _block: BlockInfo = block,
                _original: Callable[..., torch.Tensor] = original,
            ) -> torch.Tensor:
                ids = flat_expert_indices.reshape(-1, _block.top_k)
                weights = flat_expert_weights.reshape(-1, _block.top_k)
                raw_rms = torch.full(
                    ids.shape,
                    float("nan"),
                    dtype=torch.float32,
                    device=x.device,
                )
                is_probe = self._is_probe_forward(_block)
                if self.enabled and not is_probe:
                    for expert_id, expert in enumerate(_block.module.experts):
                        matches = torch.nonzero(ids == expert_id, as_tuple=False)
                        if matches.numel() == 0:
                            continue
                        token = matches[:, 0]
                        slot = matches[:, 1]
                        # Reduce immediately: upstream mutates its expert output in
                        # place when multiplying by the selected gate weight.
                        value = expert(x[token])
                        raw_rms[token, slot] = _rms_last(value)
                if is_probe:
                    pending = self._pending_probe.get(_block.layer_idx)
                    if pending is None or "router_probs" not in pending:
                        raise RuntimeError(
                            "HB vector probe dispatch fired without its gate capture"
                        )
                    self._start_probe_dispatch(_block, x, flat_expert_indices)
                try:
                    routed = _original(x, flat_expert_indices, flat_expert_weights)
                finally:
                    probe_context = self._active_probe_dispatch if is_probe else None
                    if is_probe:
                        self._active_probe_dispatch = None
                executed_routed = routed
                if self.enabled:
                    layer_idx = _block.layer_idx
                    if layer_idx in self._pending_routed:
                        raise RuntimeError(
                            "HB layer %d dispatched twice before its block completed"
                            % layer_idx
                        )
                    if is_probe:
                        if probe_context is None:
                            raise RuntimeError("HB vector probe lost its dispatch")
                        expected_experts = set(probe_context["dispatch"])
                        if probe_context["seen"] != expected_experts:
                            missing = sorted(expected_experts - probe_context["seen"])
                            raise RuntimeError(
                                "HB vector probe missed dispatched experts %s" % missing
                            )
                        if self.capture_internal_activation and (
                            probe_context["internal_seen"] != expected_experts
                        ):
                            missing = sorted(
                                expected_experts - probe_context["internal_seen"]
                            )
                            raise RuntimeError(
                                "HB internal probe missed dispatched experts %s" % missing
                            )
                        raw_vector = probe_context["raw"]
                        if not torch.isfinite(raw_vector).all():
                            raise RuntimeError(
                                "HB vector probe did not capture every selected vector"
                            )
                        raw_rms = _rms_last(raw_vector)
                        reconstructed = (
                            raw_vector * weights.float().unsqueeze(-1)
                        ).sum(dim=1)
                        routed_fp32 = routed.float()
                        reconstruction_error = (
                            (reconstructed - routed_fp32).abs().amax(dim=-1)
                        )
                        routed_scale = routed_fp32.abs().amax(dim=-1).clamp_min(1.0)
                        tolerance = (
                            self.reconstruction_atol
                            + self.reconstruction_rtol * routed_scale
                        )
                        if bool((reconstruction_error > tolerance).any()):
                            raise RuntimeError(
                                "HB vector probe fp32 routed reconstruction failed: "
                                "max error %.3e, max tolerance %.3e"
                                % (
                                    float(reconstruction_error.max()),
                                    float(tolerance.max()),
                                )
                            )
                        self._pending_probe[layer_idx].update(
                            {
                                "input_hidden": x.detach().to(
                                    dtype=torch.float16, copy=True
                                ),
                                "selected_expert_raw": raw_vector.detach(),
                                "routed_reconstruction_max_abs_error": (
                                    reconstruction_error.detach()
                                ),
                            }
                        )
                        if self.capture_internal_activation:
                            for name in self._INTERNAL_METRICS:
                                value = probe_context[name]
                                if not torch.isfinite(value).all():
                                    raise RuntimeError(
                                        "HB internal probe did not fill every selected slot"
                                    )
                                self._pending_probe[layer_idx][name] = value.detach()
                    if is_probe and self.intervention is not None:
                        result = self.intervention.apply(
                            routed,
                            raw_vector,
                            ids,
                            weights,
                            n_action_steps=self.n_action_steps,
                            pair_id=self.intervention_pair_id,
                            draw_id=self.intervention_draw_id,
                            arm=self.intervention_arm,
                            drop_policy=self.intervention_drop_policy,
                            internal_l2_flat=(
                                None
                                if not self.capture_internal_activation
                                else probe_context["internal_l2"]
                            ),
                        )
                        executed_routed = result.executed_routed_flat
                        self._pending_probe[layer_idx].update(
                            {
                                "original_routed": result.original_routed,
                                "executed_routed": result.executed_routed,
                                "intervention_delta": result.intervention_delta,
                                "dropped_slot": result.dropped_slot,
                                "dropped_expert_id": result.dropped_expert_id,
                            }
                        )
                    self._pending_routed[layer_idx] = executed_routed.detach()
                    self._pending_expert_id[layer_idx] = ids.detach()
                    self._pending_expert_weight[layer_idx] = weights.detach()
                    self._pending_expert_raw_rms[layer_idx] = raw_rms.detach()
                    self._pending_weight_error[layer_idx] = (
                        weights.float().sum(dim=-1).sub(1.0).abs().detach()
                    )
                return executed_routed

            self._wrapper[layer] = traced_moe_infer
            module.moe_infer = traced_moe_infer
            if layer == self.probe_layer:
                self._handles.append(
                    module.gate.register_forward_hook(self._probe_gate_hook(block))
                )
                for expert_id, expert in enumerate(module.experts):
                    if self.capture_internal_activation:
                        self._handles.append(
                            expert.down_proj.register_forward_pre_hook(
                                self._probe_down_proj_pre_hook(block, expert_id)
                            )
                        )
                    self._handles.append(
                        expert.register_forward_hook(
                            self._probe_expert_hook(block, expert_id)
                        )
                    )
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
        self._original_moe_infer.clear()
        self._wrapper.clear()
        self._had_instance_method.clear()
        self._instance_method.clear()
        self.attached = False

    def begin(
        self,
        episode_id: int,
        control_step: int,
        query_id: int = -1,
        candidate_id: int = -1,
        intervention_pair_id: int = -1,
        intervention_draw_id: int = -1,
        intervention_arm: str = "none",
        intervention_drop_policy: str | None = None,
    ) -> None:
        if not self.attached:
            raise RuntimeError("attach the activation recorder before begin")
        if self.enabled:
            raise RuntimeError("an activation capture is already active")
        self.episode_id = int(episode_id)
        self.control_step = int(control_step)
        self.query_id = int(query_id)
        self.candidate_id = int(candidate_id)
        self.intervention_pair_id = int(intervention_pair_id)
        self.intervention_draw_id = int(intervention_draw_id)
        self.intervention_arm = str(intervention_arm)
        self.intervention_drop_policy = (
            "none"
            if self.intervention is None
            else str(
                getattr(self.intervention, "drop_policy", "none")
                if intervention_drop_policy is None
                else intervention_drop_policy
            )
        )
        if self.intervention is not None:
            if self.intervention_pair_id < 0 or self.intervention_draw_id < 0:
                raise ValueError(
                    "paired intervention requires non-negative pair/draw IDs"
                )
            if self.intervention_arm not in tuple(self.intervention.arms):
                raise ValueError("invalid paired intervention arm")
            if self.intervention_drop_policy not in tuple(self.intervention.policies):
                raise ValueError("invalid paired intervention drop policy")
        self._rows = {block.layer_idx: [] for block in self.hb}
        self._pending_routed.clear()
        self._pending_weight_error.clear()
        self._pending_shared.clear()
        self._pending_expert_id.clear()
        self._pending_expert_weight.clear()
        self._pending_expert_raw_rms.clear()
        self._pending_probe.clear()
        self._active_probe_dispatch = None
        self.enabled = True

    def cancel(self) -> None:
        self.enabled = False
        self._rows.clear()
        self._pending_routed.clear()
        self._pending_weight_error.clear()
        self._pending_shared.clear()
        self._pending_expert_id.clear()
        self._pending_expert_weight.clear()
        self._pending_expert_raw_rms.clear()
        self._pending_probe.clear()
        self._active_probe_dispatch = None

    def _shared_hook(self, block: BlockInfo):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            layer = block.layer_idx
            if layer in self._pending_shared:
                raise RuntimeError(
                    "HB layer %d produced shared output twice before block completion"
                    % layer
                )
            self._pending_shared[layer] = output.detach()
            return None

        return hook

    def _block_hook(self, block: BlockInfo):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            layer = block.layer_idx
            routed_flat = self._pending_routed.pop(layer, None)
            weight_error_flat = self._pending_weight_error.pop(layer, None)
            shared = self._pending_shared.pop(layer, None)
            expert_id_flat = self._pending_expert_id.pop(layer, None)
            expert_weight_flat = self._pending_expert_weight.pop(layer, None)
            expert_raw_rms_flat = self._pending_expert_raw_rms.pop(layer, None)
            if (
                routed_flat is None
                or weight_error_flat is None
                or shared is None
                or expert_id_flat is None
                or expert_weight_flat is None
                or expert_raw_rms_flat is None
            ):
                raise RuntimeError(
                    "incomplete HB activation capture at layer %d: routed=%s, "
                    "weights=%s, shared=%s, raw=%s"
                    % (
                        layer,
                        routed_flat is not None,
                        weight_error_flat is not None,
                        shared is not None,
                        expert_raw_rms_flat is not None,
                    )
                )
            if output.ndim != 3:
                raise RuntimeError(
                    "HB layer %d output must be [batch, suffix, hidden], got %s"
                    % (layer, tuple(output.shape))
                )
            batch, suffix, hidden = output.shape
            expected_suffix = self.n_action_steps + 1
            if suffix != expected_suffix:
                raise RuntimeError(
                    "HB layer %d expected state + %d actions, got suffix=%d"
                    % (layer, self.n_action_steps, suffix)
                )
            if routed_flat.shape != (batch * suffix, hidden):
                raise RuntimeError(
                    "HB layer %d routed shape %s does not match block output %s"
                    % (layer, tuple(routed_flat.shape), tuple(output.shape))
                )
            if shared.shape != output.shape:
                raise RuntimeError(
                    "HB layer %d shared shape %s does not match block output %s"
                    % (layer, tuple(shared.shape), tuple(output.shape))
                )
            if weight_error_flat.numel() != batch * suffix:
                raise RuntimeError(
                    "HB layer %d has %d top-k rows for %d suffix tokens"
                    % (layer, weight_error_flat.numel(), batch * suffix)
                )

            routed = routed_flat.reshape(batch, suffix, hidden)[:, 1:]
            shared_action = shared[:, 1:]
            total_action = output.detach()[:, 1:]
            weight_error = weight_error_flat.reshape(batch, suffix)[:, 1:]
            selected_id = expert_id_flat.reshape(batch, suffix, block.top_k)[:, 1:]
            selected_weight = expert_weight_flat.reshape(batch, suffix, block.top_k)[
                :, 1:
            ]
            selected_raw_rms = expert_raw_rms_flat.reshape(batch, suffix, block.top_k)[
                :, 1:
            ]
            if not torch.isfinite(selected_raw_rms).all():
                raise RuntimeError(
                    "HB layer %d did not capture every selected expert output" % layer
                )
            row = {
                "routed_rms": _rms_last(routed),
                "shared_rms": _rms_last(shared_action),
                "total_mlp_rms": _rms_last(total_action),
                "topk_weight_sum_error": weight_error,
                "selected_expert_id": selected_id,
                "selected_expert_weight": selected_weight,
                "selected_expert_raw_rms": selected_raw_rms,
            }
            if self._is_probe_forward(block):
                probe = self._pending_probe.pop(layer, None)
                required = {
                    "router_probs",
                    "input_hidden",
                    "selected_expert_raw",
                    "routed_reconstruction_max_abs_error",
                }
                if probe is None or not required.issubset(probe):
                    missing = sorted(required - set(probe or {}))
                    raise RuntimeError(
                        "incomplete HB vector probe at layer %d: missing %s"
                        % (layer, missing)
                    )
                input_hidden = probe["input_hidden"]
                raw_vector = probe["selected_expert_raw"]
                router_probs = probe["router_probs"]
                reconstruction_error = probe["routed_reconstruction_max_abs_error"]
                if input_hidden.shape != (batch * suffix, hidden):
                    raise RuntimeError("HB vector probe input-hidden shape mismatch")
                if raw_vector.shape != (
                    batch * suffix,
                    block.top_k,
                    hidden,
                ):
                    raise RuntimeError("HB vector probe raw-vector shape mismatch")
                if router_probs.shape != (batch, suffix, block.n_experts):
                    raise RuntimeError("HB vector probe router shape mismatch")
                if reconstruction_error.shape != (batch * suffix,):
                    raise RuntimeError(
                        "HB vector probe reconstruction-error shape mismatch"
                    )
                row.update(
                    {
                        "probe_input_hidden": input_hidden.reshape(
                            batch, suffix, hidden
                        )[:, 1:],
                        "probe_shared_output": shared_action.detach().to(
                            dtype=torch.float16
                        ),
                        "probe_selected_expert_raw": raw_vector.reshape(
                            batch, suffix, block.top_k, hidden
                        )[:, 1:].to(dtype=torch.float16),
                        "probe_router_probs": router_probs[:, 1:].to(
                            dtype=torch.float16
                        ),
                        "probe_routed_reconstruction_max_abs_error": (
                            reconstruction_error.reshape(batch, suffix)[:, 1:]
                        ),
                    }
                )
                if self.capture_internal_activation:
                    missing_internal = sorted(
                        set(self._INTERNAL_METRICS) - set(probe)
                    )
                    if missing_internal:
                        raise RuntimeError(
                            "incomplete HB internal metrics: missing %s"
                            % missing_internal
                        )
                    for name in self._INTERNAL_METRICS:
                        value = probe[name]
                        if value.shape != (batch * suffix, block.top_k):
                            raise RuntimeError(
                                "HB %s shape %s is incompatible"
                                % (name, tuple(value.shape))
                            )
                        row["probe_" + name] = value.reshape(
                            batch, suffix, block.top_k
                        )[:, 1:]
                optional_intervention = {
                    "original_routed",
                    "executed_routed",
                    "intervention_delta",
                    "dropped_slot",
                    "dropped_expert_id",
                }
                if self.intervention is not None:
                    if not optional_intervention.issubset(probe):
                        missing = sorted(optional_intervention - set(probe))
                        raise RuntimeError(
                            "incomplete HB intervention capture: missing %s" % missing
                        )
                    row.update({key: probe[key] for key in optional_intervention})
            self._rows[layer].append(row)
            return None

        return hook

    def _validate_complete(self) -> None:
        pending = sorted(
            set(self._pending_routed)
            | set(self._pending_weight_error)
            | set(self._pending_shared)
            | set(self._pending_expert_id)
            | set(self._pending_expert_weight)
            | set(self._pending_expert_raw_rms)
            | set(self._pending_probe)
        )
        if self._active_probe_dispatch is not None:
            raise RuntimeError("unfinished HB vector probe dispatch")
        if pending:
            raise RuntimeError("unfinished HB forwards at layers %s" % pending)
        wrong = {
            layer: len(rows)
            for layer, rows in self._rows.items()
            if len(rows) != self.expected_denoise
        }
        if wrong:
            raise RuntimeError(
                "expected %d denoise rounds per HB layer; captured %s"
                % (self.expected_denoise, wrong)
            )

    def end(self) -> HBActivationRecord:
        if not self.enabled:
            raise RuntimeError("no active activation capture")
        self.enabled = False
        self._validate_complete()

        # One device-to-host transfer for all four metrics.  Each metric is
        # [layer, denoise, batch, action] before the batch-first permutation.
        metric_tensors = []
        for key in self._METRICS:
            value = torch.stack(
                [
                    torch.stack([row[key] for row in self._rows[layer]], dim=0)
                    for layer in self.hb_layers
                ],
                dim=0,
            )
            metric_tensors.append(value.permute(2, 0, 1, 3))
        packed = (
            torch.stack(metric_tensors, dim=0)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )

        selected_id = (
            torch.stack(
                [
                    torch.stack(
                        [row["selected_expert_id"] for row in self._rows[layer]], dim=0
                    )
                    for layer in self.hb_layers
                ],
                dim=0,
            )
            .permute(2, 0, 1, 3, 4)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        selected_weight = (
            torch.stack(
                [
                    torch.stack(
                        [row["selected_expert_weight"] for row in self._rows[layer]],
                        dim=0,
                    )
                    for layer in self.hb_layers
                ],
                dim=0,
            )
            .permute(2, 0, 1, 3, 4)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        selected_raw_rms = (
            torch.stack(
                [
                    torch.stack(
                        [row["selected_expert_raw_rms"] for row in self._rows[layer]],
                        dim=0,
                    )
                    for layer in self.hb_layers
                ],
                dim=0,
            )
            .permute(2, 0, 1, 3, 4)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )

        probe_row = self._rows[self.probe_layer][self.probe_denoise]
        probe_input_hidden = (
            probe_row["probe_input_hidden"]
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        probe_shared_output = (
            probe_row["probe_shared_output"]
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        probe_selected_expert_raw = (
            probe_row["probe_selected_expert_raw"]
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        probe_router_probs = (
            probe_row["probe_router_probs"]
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        probe_reconstruction_error = (
            probe_row["probe_routed_reconstruction_max_abs_error"]
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )

        intervention_values: dict[str, np.ndarray | None] = {
            "original_routed": None,
            "executed_routed": None,
            "intervention_delta": None,
            "dropped_slot": None,
            "dropped_expert_id": None,
        }
        internal_values: dict[str, np.ndarray | None] = {
            name: None for name in self._INTERNAL_METRICS
        }
        if self.capture_internal_activation:
            for name in self._INTERNAL_METRICS:
                internal_values[name] = (
                    probe_row["probe_" + name]
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                )
        if self.intervention is not None:
            for key in ("original_routed", "executed_routed", "intervention_delta"):
                intervention_values[key] = (
                    probe_row[key].to(device="cpu", dtype=torch.float32).numpy()
                )
            intervention_values["dropped_slot"] = (
                probe_row["dropped_slot"].to(device="cpu", dtype=torch.int8).numpy()
            )
            intervention_values["dropped_expert_id"] = (
                probe_row["dropped_expert_id"]
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )

        record = HBActivationRecord(
            episode_id=self.episode_id,
            control_step=self.control_step,
            query_id=self.query_id,
            candidate_id=self.candidate_id,
            n_denoise=self.expected_denoise,
            hb_layers=np.asarray(self.hb_layers, dtype=np.int16),
            hb_routed_rms=packed[0],
            hb_shared_rms=packed[1],
            hb_total_mlp_rms=packed[2],
            hb_topk_weight_sum_error=packed[3],
            hb_selected_expert_id=selected_id,
            hb_selected_expert_weight=selected_weight,
            hb_selected_expert_raw_rms=selected_raw_rms,
            probe_hb_layer=self.probe_layer,
            probe_denoise=self.probe_denoise,
            hb_probe_input_hidden=probe_input_hidden,
            hb_probe_shared_output=probe_shared_output,
            hb_probe_selected_expert_raw=probe_selected_expert_raw,
            hb_probe_router_probs=probe_router_probs,
            hb_probe_routed_reconstruction_max_abs_error=(probe_reconstruction_error),
            intervention_pair_id=self.intervention_pair_id,
            intervention_draw_id=self.intervention_draw_id,
            intervention_arm=self.intervention_arm,
            intervention_drop_policy=(
                self.intervention_drop_policy
            ),
            hb_probe_original_routed=intervention_values["original_routed"],
            hb_probe_executed_routed=intervention_values["executed_routed"],
            hb_probe_intervention_delta=intervention_values["intervention_delta"],
            hb_probe_dropped_slot=intervention_values["dropped_slot"],
            hb_probe_dropped_expert_id=intervention_values["dropped_expert_id"],
            hb_probe_internal_l2=internal_values["internal_l2"],
            hb_probe_internal_l1=internal_values["internal_l1"],
            hb_probe_internal_signed_mean=internal_values[
                "internal_signed_mean"
            ],
            hb_probe_internal_positive_fraction=internal_values[
                "internal_positive_fraction"
            ],
            hb_probe_internal_linf=internal_values["internal_linf"],
        )
        self._rows.clear()
        return record
