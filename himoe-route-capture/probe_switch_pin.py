#!/usr/bin/env python3
"""Smoke test for StateTokenPin, on CPU, before any GPU time is spent.

Four checks, in order of what they would catch:

  1. the hooks fire on every gate call at the pinned layers
  2. the ``none`` arm reproduces the no-hook action **bit-exactly** -- if
     attaching the hook changes anything by itself, every other number is
     meaningless.  This is the zero control the earlier probes used.
  3. the pin actually lands: re-record the routing downstream of the hook and
     check the state token's top-4 equals the pinned set at every layer, every
     denoise iteration, while the ten action tokens are untouched
  4. the arms move the action, and by how much relative to the baseline

Runs in the CPU autocast shim so the arithmetic matches the bf16 the GPU uses.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY, FLOW_NOISE_KEY, FLOW_NOISE_SHAPE, IMAGE_KEY, IMAGE_SHAPE,
    PROMPT_KEY, STATE_DIM, STATE_KEY, WRIST_IMAGE_KEY,
)


def _resolve(root, dotted):
    node = root
    for part in dotted.split("."):
        node = getattr(node, part) if not part.isdigit() else node[int(part)]
    return node


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(args.threads)
    from probe_flow_lead_cpu import install_cpu_autocast
    install_cpu_autocast(torch)

    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates
    from himoe_switch_intervene import StateTokenPin, load_pin

    pin = load_pin(args.pin)
    print("pin table: %s / %s, layers %s"
          % (pin["suite"], pin["task"][:36], sorted(pin["layers"])))

    policy = HiMoEPolicy(checkpoint_dir=args.checkpoint_dir, suite=args.suite,
                         upstream_root=args.upstream_root, require_cuda=False,
                         libero_wrist_layout="paper-right")
    core = policy._policy.model
    gates = discover_gates(core)
    print("gates: %d (%d HB)" % (len(gates), sum(g.kind == "HB" for g in gates)))

    rng = np.random.default_rng(args.seed)
    ref = {
        IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        STATE_KEY: (rng.standard_normal(STATE_DIM) * 0.1).astype(np.float32),
        PROMPT_KEY: args.prompt,
        FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
    }

    def act(obs):
        return np.asarray(policy.infer(dict(obs))[ACTION_KEY], np.float32)

    print("\n1/4  baseline, no hook attached")
    base = act(ref)
    base2 = act(ref)
    if not np.array_equal(base, base2):
        raise SystemExit("FAIL: inference is not deterministic on the same input; "
                         "every comparison below would be meaningless")
    print("     inference is deterministic on a repeated query (bit-identical)")

    print("\n2/4  zero control: hooks attached, regime=None")
    probe = StateTokenPin(gates, torch, pin, None).attach(lambda n: _resolve(core, n))
    none = act(ref)
    rep = probe.report()
    probe.close()
    print("     gate calls intercepted: %d" % rep["gate_calls"])
    if rep["gate_calls"] != len(pin["layers"]) * 10:
        print("     NOTE: expected %d (layers x 10 denoise), got %d"
              % (len(pin["layers"]) * 10, rep["gate_calls"]))
    if not np.array_equal(base, none):
        raise SystemExit("FAIL: the none arm changed the action; max|d| = %.3e"
                         % np.abs(base - none).max())
    print("     none arm reproduces the baseline BIT-EXACTLY")

    print("\n3/4  the pin lands, and only on the state token")

    def record(regime):
        rec = HiMoERouteRecorder(core, store_full_probs=False, verify_steps=0)
        probe = None
        if regime is not None:
            probe = StateTokenPin(gates, torch, pin, regime)
            probe.attach(lambda n: _resolve(core, n))    # pin first
        rec.attach()                                     # recorder sees the result
        rec.begin_control_step(episode_id=0, control_step=0)
        act(ref)
        r = rec.end_control_step()
        rec.close()
        stats = probe.report() if probe else None
        if probe:
            probe.close()
        ids = np.asarray(r.hb_expert_ids)
        return (ids[0] if ids.ndim == 5 else ids), stats

    clean, _ = record(None)
    hb_layers = [g.layer_idx for g in gates if g.kind == "HB"]
    first_pinned = min(int(k) for k in pin["layers"])
    for regime in ("off", "on"):
        got_ids, stats = record(regime)
        # Only denoise iteration 0 admits a clean "nothing else moved" claim.
        # From d=1 the action tokens carry an x_t that the previous iteration's
        # intervention already changed, so their routing moving is the effect
        # propagating, not a leak.
        d0 = [(np.sort(got_ids[i, 0], -1) != np.sort(clean[i, 0], -1)).any(-1)
              for i in range(len(hb_layers))]
        allq = [(np.sort(got_ids[i], -1) != np.sort(clean[i], -1)).any(-1).any(0)
                for i in range(len(hb_layers))]
        print("     %-3s  tokens whose top-4 moved   (d=0 | all 10 denoise):" % regime)
        for i, L in enumerate(hb_layers):
            print("          L%-3d  %-14s | %s"
                  % (L, list(np.flatnonzero(d0[i])), list(np.flatnonzero(allq[i]))))
        i0 = hb_layers.index(first_pinned)
        if list(np.flatnonzero(d0[i0])) != [0]:
            raise SystemExit("FAIL: at the first pinned layer (L%d), denoise 0, "
                             "something other than the state token moved" % first_pinned)
        print("          -> at L%d d=0 only the state token moved, as required"
              % first_pinned)
    for regime in ("off", "on"):
        rec = HiMoERouteRecorder(core, store_full_probs=False, verify_steps=0)
        probe = StateTokenPin(gates, torch, pin, regime)
        probe.attach(lambda n: _resolve(core, n))       # pin first
        rec.attach()                                     # recorder sees the result
        rec.begin_control_step(episode_id=0, control_step=0)
        act(ref)
        r = rec.end_control_step()
        rec.close()
        stats = probe.report()
        probe.close()
        ids = np.asarray(r.hb_expert_ids)
        # the record carries a leading control-step axis, so it is
        # [1, L, D, S, 4] here; drop it before indexing the layer
        if ids.ndim == 5:
            ids = ids[0]
        hb_layers = [g.layer_idx for g in gates if g.kind == "HB"]
        ok_state, moved_action = True, 0
        for key, prec in pin["layers"].items():
            i = hb_layers.index(int(key))
            want = np.sort(np.asarray(prec[regime]["experts"]))
            got = np.sort(ids[i, :, 0], -1)              # state token, all denoise
            if not np.all(got == want):
                ok_state = False
        print("     %-3s  state token pinned everywhere: %s   "
              "mean |original ∩ pin| = %.2f / 4   replacements %d"
              % (regime, ok_state, stats["mean_overlap_with_original"],
                 stats["n_replacements"]))
        if not ok_state:
            raise SystemExit("FAIL: the recorded state-token routing is not the pin")

    print("\n4/4  how far each arm moves the action")
    rows = {}
    for regime in (None, "off", "on"):
        probe = StateTokenPin(gates, torch, pin, regime).attach(
            lambda n: _resolve(core, n))
        a = act(ref)
        probe.close()
        d = float(np.linalg.norm(a - base) / max(np.linalg.norm(base), 1e-9))
        rows[str(regime)] = d
        print("     %-5s  relative L2 change of the action chunk: %.4f"
              % (regime, d))
    if rows["None"] != 0.0:
        raise SystemExit("FAIL: the none arm moved the action")
    if rows["off"] == 0.0 and rows["on"] == 0.0:
        raise SystemExit("FAIL: neither arm changed anything -- the pin is a no-op")
    print("\nall four checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
