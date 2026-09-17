"""Pin the state token's HB routing to a constant, to test whether the gate does
anything functionally.

The only input-dependent routing left in HiMoE is a bimodal on/off switch on the
state token (suffix index 0) at HB layers 2-5, keyed on the gripper -- the ten
action tokens carry a fixed positional code and the back block never leaves
uniform.  So the causal question is answered by replacing *that one token's*
routing with a constant and rerunning the rollout: if success does not move, the
model's only adaptive routing is functionally inert.

This is a forward hook on the gate, not a mirror of it.  `register_forward_hook`
lets a hook return a replacement for the module's output, and `MoEGate_load_bal`
returns `(topk_idx, topk_weight, aux_loss)`, so the hook rewrites row 0 of the
first two and leaves everything else -- every action token, every other layer,
the shared expert, the whole back block -- untouched.

Ordering matters if `HiMoERouteRecorder` is attached at the same time.  Hooks fire
in registration order and each sees the output as revised by the ones before it,
so attach the recorder FIRST to record the routing the model would have produced,
or attach it SECOND to record what was actually executed.  The recorder's own
correctness check compares its recomputed top-k against the gate's returned
`topk_idx`, which an intervention deliberately breaks, so pass `verify_steps=0`
when recording downstream of a pin.

`pin_weights` must sum to 1: `norm_topk_prob` is true for HB and top_k is 4, so
the gate itself always emits weights that sum to 1, and the moe_infer branch
multiplies each expert's output by its weight without renormalising again.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np


class StateTokenPin:
    """Replace the state token's top-4 and weights at chosen HB layers.

    ``regime`` of ``None`` attaches the hooks but changes nothing, which is the
    zero control every arm is checked against: the action it produces must be
    bit-identical to the action with no hook attached at all.
    """

    def __init__(self, gates, torch, pin: dict, regime: str | None,
                 suffix_index: int = 0) -> None:
        self.torch = torch
        self.regime = regime
        self.suffix_index = suffix_index
        self.handles: list[Any] = []
        self.calls = 0
        self.overlap: list[int] = []      # |original top-4 ∩ pin| per call
        self.layers: dict[int, Any] = {}

        by_layer = {g.layer_idx: g for g in gates if g.kind == "HB"}
        for key, rec in pin["layers"].items():
            layer = int(key)
            if regime is not None and regime not in rec:
                raise ValueError("pin has no %r regime for layer %d" % (regime, layer))
            if layer not in by_layer:
                raise ValueError("layer %d is not an HB gate" % layer)
            self.layers[layer] = by_layer[layer]
        if not self.layers:
            raise ValueError("pin table selected no layers")
        self.pin = pin

    def attach(self, resolve) -> "StateTokenPin":
        for layer, gate in sorted(self.layers.items()):
            module = resolve(gate.name)
            self.handles.append(module.register_forward_hook(self._make(layer)))
        return self

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make(self, layer: int):
        torch = self.torch
        rec = self.pin["layers"][str(layer)]

        def hook(module, args, output):
            self.calls += 1
            if self.regime is None:
                return None
            topk_idx, topk_weight, aux = output
            bsz, seq_len, _ = args[0].shape
            k = topk_idx.shape[-1]
            idx = topk_idx.reshape(bsz, seq_len, k).clone()
            w = topk_weight.reshape(bsz, seq_len, k).clone()

            want = rec[self.regime]
            pin_idx = torch.as_tensor(want["experts"], dtype=idx.dtype,
                                      device=idx.device)
            pin_w = torch.as_tensor(want["weights"], dtype=w.dtype, device=w.device)

            before = set(int(v) for v in idx[0, self.suffix_index].tolist())
            self.overlap.append(len(before & set(want["experts"])))

            idx[:, self.suffix_index, :] = pin_idx
            w[:, self.suffix_index, :] = pin_w
            return idx.reshape(topk_idx.shape), w.reshape(topk_weight.shape), aux

        return hook

    def report(self) -> dict:
        ov = np.asarray(self.overlap, float)
        return {"regime": self.regime, "layers": sorted(self.layers),
                "gate_calls": self.calls,
                "mean_overlap_with_original": float(ov.mean()) if ov.size else None,
                "n_replacements": int(ov.size)}


def load_pin(path: str | pathlib.Path) -> dict:
    pin = json.loads(pathlib.Path(path).read_text())
    for key, rec in pin["layers"].items():
        for regime in ("on", "off"):
            if regime not in rec:
                continue
            w = np.asarray(rec[regime]["weights"], float)
            if abs(w.sum() - 1.0) > 1e-5:
                raise ValueError("layer %s %s weights sum to %.6f, not 1"
                                 % (key, regime, w.sum()))
            if len(set(rec[regime]["experts"])) != 4:
                raise ValueError("layer %s %s does not name 4 distinct experts"
                                 % (key, regime))
    return pin
