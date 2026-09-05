#!/usr/bin/env python
"""Route-recorder server with mid-flow latent injection (E1-B) and an optional
flow-step override (E5/H8).

Wraps -- never edits -- `serve_with_recorder.py`.  Two patch points, both of
which are imported *inside* its main(), so patching the bridge module before
calling main() is enough:

  create_policy  -> lets us set model.config.num_steps before the recorder and
                    the zarr writers read it (F override must happen there or
                    the stores get sized for the wrong F).
  PolicyServer   -> constructed last, after `policy.infer` is installed, so it
                    is the right moment to wrap infer and denoise_step.

Wire additions (both optional; absent = ordinary behaviour):

  request  inject/step   int    denoise step index f* to perturb, 0-based
           inject/delta  f32    [n_action_steps, max_action_dim] added to x_t
                                just before the velocity of step f* is computed
  response inject/x_norm_before  float  ||x_t|| before the addition, so the
                                        client can report a *relative* epsilon
           inject/applied         bool   injection actually fired

Determinism contract that makes E1-B self-checking: with the same flow noise,
routing at steps f < f* is bit-identical to the uninjected run, so the analyzer
can assert D_f == 0 there and only interpret f >= f*.

Usage (env carries our options; everything else forwards to serve_with_recorder):
  HIMOE_FLOW_STEPS=20 python serve_latent_inject.py --port 9319 --gpu 6 ...
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")

INJECT_STEP_KEY = "inject/step"
INJECT_DELTA_KEY = "inject/delta"
INJECT_NORM_RESPONSE_KEY = "inject/x_norm_before"
INJECT_APPLIED_RESPONSE_KEY = "inject/applied"

# Per-request injection state.  The bridge server handles one connection at a
# time on a single asyncio loop, so a module-level slot is safe here; it is
# reset at the top of every infer so a crashed request cannot leak into the
# next one.
_STATE = {"step": None, "delta": None, "seen": 0, "applied": False,
          "x_norm": float("nan")}


def _install(policy) -> None:
    import torch

    model = policy._policy.model
    original_denoise = model.denoise_step

    def denoise_step(*args, **kwargs):
        # bound method: (state, prefix_pad_masks, prefix_att_masks,
        #                past_key_values, data_mask, x_t, timestep)
        x_t = kwargs.get("x_t", args[5] if len(args) > 5 else None)
        step = _STATE["step"]
        if step is not None and _STATE["seen"] == step and x_t is not None:
            _STATE["x_norm"] = float(torch.linalg.vector_norm(x_t).item())
            delta = torch.as_tensor(_STATE["delta"], dtype=x_t.dtype,
                                    device=x_t.device).reshape(x_t.shape)
            # in-place: sample_actions keeps updating this same tensor object
            # (x_t += dt * v), so the perturbation persists for the rest of the
            # flow exactly as a genuine state perturbation would.
            x_t.add_(delta)
            _STATE["applied"] = True
        _STATE["seen"] += 1
        return original_denoise(*args, **kwargs)

    model.denoise_step = denoise_step

    inner_infer = policy.infer

    def infer(observation):
        step = observation.pop(INJECT_STEP_KEY, None)
        delta = observation.pop(INJECT_DELTA_KEY, None)
        _STATE.update(step=None, delta=None, seen=0, applied=False,
                      x_norm=float("nan"))
        if step is not None:
            if delta is None:
                raise ValueError("inject/step given without inject/delta")
            step = int(step)
            if step < 0:
                raise ValueError("inject/step must be >= 0")
            delta = np.asarray(delta, dtype=np.float32)
            if not np.isfinite(delta).all():
                raise ValueError("inject/delta must be finite")
            _STATE.update(step=step, delta=delta)
        response = inner_infer(observation)
        if step is not None:
            if not _STATE["applied"]:
                raise RuntimeError(
                    "inject/step %d never fired (%d denoise steps ran)"
                    % (step, _STATE["seen"]))
            response[INJECT_NORM_RESPONSE_KEY] = float(_STATE["x_norm"])
            response[INJECT_APPLIED_RESPONSE_KEY] = True
        return response

    policy.infer = infer
    meta = getattr(policy, "metadata", None)
    if isinstance(meta, dict):
        meta["latent_injection"] = {
            "step_key": INJECT_STEP_KEY, "delta_key": INJECT_DELTA_KEY,
            "norm_response_key": INJECT_NORM_RESPONSE_KEY,
            "semantics": "x_t += delta immediately before the velocity of "
                         "denoise step f* is computed; steps < f* are "
                         "bit-identical to the uninjected run",
        }
    print("[inject] latent injection armed on model.denoise_step", flush=True)


def main() -> int:
    import himoe_libero_bridge.server as bridge_server

    flow_steps = os.environ.get("HIMOE_FLOW_STEPS")
    original_create = bridge_server.create_policy
    original_server = bridge_server.PolicyServer

    def create_policy(*args, **kwargs):
        policy = original_create(*args, **kwargs)
        if flow_steps:
            cfg = policy._policy.model.config
            old = int(cfg.num_steps)
            cfg.num_steps = int(flow_steps)
            meta = getattr(policy, "metadata", None)
            if isinstance(meta, dict):
                meta["flow_steps"] = int(flow_steps)
                meta["flow_steps_overridden_from"] = old
            print("[inject] flow steps %d -> %s" % (old, flow_steps), flush=True)
        return policy

    class InjectingPolicyServer(original_server):
        def __init__(self, policy, host, port, backend):
            _install(policy)
            super().__init__(policy, host, port, backend)

    bridge_server.create_policy = create_policy
    bridge_server.PolicyServer = InjectingPolicyServer

    import serve_with_recorder as swr
    return swr.main()


if __name__ == "__main__":
    raise SystemExit(main())
