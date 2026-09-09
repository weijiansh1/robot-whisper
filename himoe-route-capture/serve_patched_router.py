"""Force one control step's HB routing to come from somewhere else.

The behavioural ablations so far (`serve_ablated_router.py`, `serve_ablated_branch.py`)
perturb every control step and every denoising round at once.  The routing outcome
signal, however, lives in a two-step window: at control steps 11-12 the routing
state decodes success at 0.675 after residualising against the full physical world
state, and from step 13 on it is a pure state readout.  A perturbation smeared over
the whole episode cannot test that window -- closed-loop replanning absorbs it.

This server patches a single control step.  Routing is captured into a named slot
during one rollout and replayed into a later rollout, so the donor routing always
comes from a live run in this same process rather than from a stored capture that
may not reproduce bit-exactly (see run_objstate_capture.sh: identical seeds gave
27/64 against the original 23/64).

Only HB gates are touched.  AS routing is constant on LIBERO -- `data_mask` never
varies within a suite -- so there is nothing there to transplant.

Observation keys, all popped before the policy sees them:

  patch/record   slot name; store this call's HB routing under it
  patch/apply    slot name; override this call's HB routing from it
  patch/random   truthy; override with a uniformly random top-4, keeping the
                 router's own normalised weights (same semantics as
                 serve_ablated_router.py --mode random, so the two are comparable)

`patch/apply` and `patch/random` are mutually exclusive.  Recording and applying in
the same call is allowed and is how the self-patch identity check is run: the stored
slot must reproduce bit-exactly.

Response keys: patch/recorded_sites, patch/applied_sites.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")

RECORD_KEY = "patch/record"
APPLY_KEY = "patch/apply"
RANDOM_KEY = "patch/random"
RETURN_KEY = "patch/return_routing"
RECORDED_SITES_KEY = "patch/recorded_sites"
APPLIED_SITES_KEY = "patch/applied_sites"
ROUTING_IDX_KEY = "patch/routing_idx"
ROUTING_WEIGHT_KEY = "patch/routing_weight"


class RoutePatcher:
    """Hooks every HB gate; records or overrides (topk_idx, topk_weight)."""

    def __init__(self, core, n_experts: int = 32, top_k: int = 4):
        from himoe_router_recorder import discover_gates

        self.gates = [g for g in discover_gates(core) if g.kind == "HB"]
        if not self.gates:
            raise RuntimeError("no HB gates found; wrong module passed in")
        for gate in self.gates:
            if gate.n_experts != n_experts or gate.top_k != top_k:
                raise RuntimeError(
                    "unexpected HB gate %s: experts=%d top_k=%d"
                    % (gate.name, gate.n_experts, gate.top_k)
                )
        self.n_experts = n_experts
        self.top_k = top_k
        self._first_layer = self.gates[0].layer_idx
        self._handles = []
        self.slots: dict[str, dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]] = {}

        self._denoise = -1
        self._record_into: str | None = None
        self._apply_from: str | None = None
        self._staging: dict | None = None
        self._randomise = False
        self._generator: torch.Generator | None = None
        self.recorded_sites = 0
        self.applied_sites = 0

    def attach(self) -> "RoutePatcher":
        for gate in self.gates:
            self._handles.append(gate.module.register_forward_hook(self._make_hook(gate)))
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_call(self, record: str | None, apply: str | None, randomise: bool) -> None:
        if apply is not None and randomise:
            raise ValueError("patch/apply and patch/random are mutually exclusive")
        if apply is not None and apply not in self.slots:
            raise KeyError("unknown patch slot %r; recorded slots: %s"
                           % (apply, sorted(self.slots)))
        self._denoise = -1
        self._record_into = record
        self._apply_from = apply
        self._randomise = bool(randomise)
        self.recorded_sites = 0
        self.applied_sites = 0
        # Record into a staging dict and only commit it at end_call.  Clearing
        # self.slots[record] here would destroy the donor whenever record == apply,
        # which is exactly the self-patch identity check: the hook would then read
        # back whatever it had just written, reproduce it bit-exactly by
        # construction, and report a full patch that was a no-op -- while also
        # losing the phase-A recording for that seed.
        self._staging = {} if record is not None else None

    def pack(self, slot: str) -> tuple[np.ndarray, np.ndarray]:
        """A slot as dense arrays [L, D, S, K], layers in gate execution order.

        The client needs the routing itself, not just a handle, to ask whether a
        transplanted donor lies inside the cloud the policy can reach at this state
        by resampling its own flow noise.
        """
        entries = self.slots[slot]
        layers = [g.layer_idx for g in self.gates]
        rounds = sorted({d for _, d in entries})
        idx = np.stack([
            np.stack([entries[(layer, d)][0].detach().cpu().numpy() for d in rounds])
            for layer in layers
        ])
        weight = np.stack([
            np.stack([entries[(layer, d)][1].detach().float().cpu().numpy() for d in rounds])
            for layer in layers
        ])
        return idx.astype(np.uint8), weight.astype(np.float32)

    def end_call(self) -> tuple[int, int]:
        if self._record_into is not None:
            self.slots[self._record_into] = self._staging
        self._staging = None
        self._record_into = None
        self._apply_from = None
        self._randomise = False
        return self.recorded_sites, self.applied_sites

    def _make_hook(self, gate):
        def hook(module, args, output):
            if self._record_into is None and self._apply_from is None and not self._randomise:
                return None
            if gate.layer_idx == self._first_layer:
                self._denoise += 1
            key = (gate.layer_idx, self._denoise)
            topk_idx, topk_weight, aux_loss = output

            if self._record_into is not None:
                self._staging[key] = (
                    topk_idx.detach().clone(),
                    topk_weight.detach().clone(),
                )
                self.recorded_sites += 1

            if self._apply_from is not None:
                stored = self.slots[self._apply_from].get(key)
                if stored is None:
                    raise KeyError(
                        "slot %r has no routing for layer %d denoise %d; the donor "
                        "rollout ran a different number of denoising rounds"
                        % (self._apply_from, gate.layer_idx, self._denoise)
                    )
                donor_idx, donor_weight = stored
                if donor_idx.shape != topk_idx.shape:
                    raise RuntimeError(
                        "donor routing at layer %d denoise %d has shape %s but this "
                        "call needs %s; suffix length changed between rollouts"
                        % (gate.layer_idx, self._denoise, tuple(donor_idx.shape),
                           tuple(topk_idx.shape))
                    )
                self.applied_sites += 1
                return (
                    donor_idx.to(topk_idx.device, topk_idx.dtype),
                    donor_weight.to(topk_weight.device, topk_weight.dtype),
                    aux_loss,
                )

            if self._randomise:
                n_token = topk_idx.shape[0]
                if self._generator is None:
                    self._generator = torch.Generator(device=topk_idx.device)
                    self._generator.manual_seed(0)
                scores = torch.rand(
                    (n_token, self.n_experts),
                    generator=self._generator,
                    device=topk_idx.device,
                )
                random_idx = scores.topk(self.top_k, dim=-1, sorted=False).indices
                self.applied_sites += 1
                return (random_idx.to(topk_idx.dtype), topk_weight, aux_loss)

            return None

        return hook


def _as_slot(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="released-left")
    ap.add_argument("--out", help="directory for the shutdown summary")
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    policy = create_policy(
        "himoe", args.checkpoint_dir, args.upstream_root, args.gpu,
        args.suite, args.libero_wrist_layout,
    )

    core = policy._policy.model
    patcher = RoutePatcher(core).attach()
    print("HB gates:", [g.layer_idx for g in patcher.gates], flush=True)

    stats = {"calls": 0, "recorded_calls": 0, "applied_calls": 0, "random_calls": 0}
    inner = policy.infer

    def infer(observation):
        record = _as_slot(observation.pop(RECORD_KEY, None))
        apply = _as_slot(observation.pop(APPLY_KEY, None))
        randomise = bool(observation.pop(RANDOM_KEY, False))
        return_routing = bool(observation.pop(RETURN_KEY, False))
        if return_routing and record is None:
            raise ValueError("patch/return_routing needs patch/record to name a slot")
        patcher.begin_call(record, apply, randomise)
        try:
            response = inner(observation)
        finally:
            recorded, applied = patcher.end_call()
        stats["calls"] += 1
        stats["recorded_calls"] += int(record is not None)
        stats["applied_calls"] += int(apply is not None)
        stats["random_calls"] += int(randomise)
        response = dict(response)
        response[RECORDED_SITES_KEY] = int(recorded)
        response[APPLIED_SITES_KEY] = int(applied)
        if return_routing:
            idx, weight = patcher.pack(record)
            response[ROUTING_IDX_KEY] = idx
            response[ROUTING_WEIGHT_KEY] = weight
        return response

    policy.infer = infer
    policy.metadata["route_patcher"] = "serve_patched_router"
    policy.metadata["route_patcher_hb_layers"] = [g.layer_idx for g in patcher.gates]

    def shutdown():
        patcher.close()
        if args.out:
            out = pathlib.Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "patch_server_summary.json").write_text(
                json.dumps({**stats, "slots": sorted(patcher.slots)}, indent=2)
            )
        print("patch server stats:", stats, flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (shutdown(), sys.exit(0)))

    print("serving on ws://127.0.0.1:%d" % args.port, flush=True)
    try:
        PolicyServer(policy, "127.0.0.1", args.port, "himoe").serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
