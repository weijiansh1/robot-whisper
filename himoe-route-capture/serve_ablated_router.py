"""Behavioural test: does the HB router's choice change what the policy does?

A forward hook on each HB gate REPLACES (topk_idx, topk_weight, aux_loss) with
an alternative selection, so the ablation is exercised in the real forward pass
and the policy is then evaluated by task success -- not by an offline norm.

  --mode none      untouched (control)
  --mode random    uniformly random top-4, weights renormalised from the router's
                   own probabilities for those experts
  --mode worst     the four lowest-probability experts
  --mode uniform   the router's own top-4, but all weights forced to 1/4

AS-MoE gates are never touched.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="checkpoint-right")
    ap.add_argument("--mode", choices=["none", "random", "worst", "uniform"],
                    required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root,
                           args.gpu, args.suite, args.libero_wrist_layout)
    expert_model = policy._policy.model.paligemma_with_expert.gemma_expert
    hb = [(i, l.mlp) for i, l in enumerate(expert_model.layers)
          if type(l.mlp).__name__ == "HBMoE"]
    print("HB layers to ablate:", [i for i, _ in hb], flush=True)
    print("mode =", args.mode, flush=True)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    counts = {"calls": 0}

    def make_hook(mlp):
        def hook(gate, inputs, output):
            if args.mode == "none":
                return None
            idx, w, aux = output
            x = inputs[0]
            k = gate.top_k
            with torch.no_grad():
                logits = F.linear(x.reshape(-1, x.shape[-1]), gate.weight, None)
                probs = logits.softmax(-1)
                if args.mode == "random":
                    r = torch.rand(probs.shape, generator=gen, device=probs.device)
                    new_idx = r.topk(k, dim=-1).indices
                elif args.mode == "worst":
                    new_idx = (-probs).topk(k, dim=-1).indices
                else:  # uniform weights, same experts
                    new_idx = idx
                p = probs.gather(-1, new_idx)
                if args.mode == "uniform":
                    new_w = torch.full_like(p, 1.0 / k)
                else:
                    new_w = p / (p.sum(-1, keepdim=True) + 1e-20)
            counts["calls"] += 1
            return new_idx, new_w.to(w.dtype), aux

        return hook

    for _, mlp in hb:
        mlp.gate.register_forward_hook(make_hook(mlp))

    policy.metadata["hb_router_ablation"] = args.mode
    inner = policy.infer

    def infer(observation):
        out = inner(observation)
        if counts["calls"] and counts["calls"] % 4000 == 0:
            print("ablated %d gate calls" % counts["calls"], flush=True)
        return out

    policy.infer = infer
    print("serving on ws://127.0.0.1:%d" % args.port, flush=True)
    PolicyServer(policy, "127.0.0.1", args.port, "himoe").serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
