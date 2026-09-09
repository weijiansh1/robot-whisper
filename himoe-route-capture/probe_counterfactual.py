"""Does the HB router's choice actually matter?

Captures the real inputs to every HBMoE block during live inference, then for
each one recomputes the block output under alternative routings:

  true    the router's own top-4              (ground truth)
  rand    a uniformly random top-4, same weights normalisation
  worst   the four LOWEST-probability experts
  allavg  every expert, weight 1/32
  shared  shared expert only (routed branch removed)

If swapping the router's pick for a random pick barely moves the block output,
the router is not doing meaningful work regardless of how balanced its load is.

Run in the model env; drive it with the normal LIBERO client.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")


def moe_forward(mlp, x, idx, weights):
    """Replicate HBMoE.moe_infer for an arbitrary (idx, weights) assignment."""
    flat = x.reshape(-1, x.shape[-1])
    out = torch.zeros_like(flat, dtype=torch.float32)
    for slot in range(idx.shape[-1]):
        e_ids = idx[..., slot].reshape(-1)
        w = weights[..., slot].reshape(-1, 1)
        for e in torch.unique(e_ids):
            m = e_ids == e
            out[m] += mlp.experts[int(e)](flat[m]).float() * w[m]
    return out.reshape(x.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="checkpoint-right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-control-steps", type=int, default=24)
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root,
                           args.gpu, args.suite, args.libero_wrist_layout)
    core = policy._policy.model
    expert_model = core.paligemma_with_expert.gemma_expert

    hb = [(i, l.mlp) for i, l in enumerate(expert_model.layers)
          if type(l.mlp).__name__ == "HBMoE"]
    print("HB layers:", [i for i, _ in hb], flush=True)

    rng = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    state = {"n": 0, "done": False}

    def make_hook(layer_idx, mlp):
        def hook(mod, inputs, output):
            if state["done"]:
                return None
            x = inputs[0]
            with torch.no_grad():
                idx, w, _ = mlp.gate(x)
                bsz, seq, _ = x.shape
                k = idx.shape[-1]
                idx = idx.reshape(bsz, seq, k)
                w = w.reshape(bsz, seq, k)

                logits = torch.nn.functional.linear(
                    x.reshape(-1, x.shape[-1]), mlp.gate.weight, None)
                probs = logits.softmax(-1).reshape(bsz, seq, -1)

                y_true = moe_forward(mlp, x, idx, w)
                shared = mlp.shared_experts(x).float()

                # random top-4, renormalised over the same router probabilities
                r = torch.rand(bsz, seq, probs.shape[-1], generator=rng,
                               device=x.device)
                ridx = r.topk(k, dim=-1).indices
                rp = probs.gather(-1, ridx)
                rw = rp / rp.sum(-1, keepdim=True)
                y_rand = moe_forward(mlp, x, ridx, rw)

                # the four least-preferred experts
                widx = (-probs).topk(k, dim=-1).indices
                wp = probs.gather(-1, widx)
                y_worst = moe_forward(mlp, x, widx, wp / wp.sum(-1, keepdim=True))

                # every expert, equal weight
                aidx = torch.arange(probs.shape[-1], device=x.device)
                aidx = aidx.view(1, 1, -1).expand(bsz, seq, -1)
                aw = torch.full_like(aidx, 1, dtype=torch.float32) / probs.shape[-1]
                y_all = moe_forward(mlp, x, aidx, aw)

                def rel(a, b):
                    return float((a - b).norm() / b.norm())

                rows.append({
                    "layer": layer_idx,
                    "x_norm": float(x.float().norm()),
                    "routed_norm": float(y_true.norm()),
                    "shared_norm": float(shared.norm()),
                    "block_out_norm": float((y_true + shared).norm()),
                    "rel_rand_vs_true": rel(y_rand, y_true),
                    "rel_worst_vs_true": rel(y_worst, y_true),
                    "rel_allavg_vs_true": rel(y_all, y_true),
                    # how much the whole block output moves, which is what the
                    # residual stream actually sees
                    "block_rel_rand": rel(y_rand + shared, y_true + shared),
                    "block_rel_worst": rel(y_worst + shared, y_true + shared),
                    "top1_prob": float(probs.max(-1).values.mean()),
                    "entropy": float(-(probs.clamp_min(1e-12) *
                                       probs.clamp_min(1e-12).log()).sum(-1).mean()),
                })
            return None

        return hook

    handles = [mlp.register_forward_hook(make_hook(i, mlp)) for i, mlp in hb]

    inner = policy.infer

    def infer(observation):
        response = inner(observation)
        state["n"] += 1
        if state["n"] >= args.max_control_steps and not state["done"]:
            state["done"] = True
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(out, "w"))
            print("wrote %d rows from %d control steps -> %s"
                  % (len(rows), state["n"], out), flush=True)
        return response

    policy.infer = infer
    print("serving on ws://127.0.0.1:%d" % args.port, flush=True)
    PolicyServer(policy, "127.0.0.1", args.port, "himoe").serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
