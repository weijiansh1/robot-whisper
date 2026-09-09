"""HiMoE CALVIN-D policy server with the HB-MoE ablations attached.

Ports the LIBERO ablation harness (serve_ablated_router.py / serve_ablated_branch.py)
to the CALVIN evaluator, so the same question can be asked on a second benchmark:

    router_random  keep the MoE block, replace the top-4 choice with a uniform
                   random draw          -> does *which* experts matter?
    shared_off     zero the always-on shared branch   -> routed branch alone
    routed_off     zero the routed branch             -> shared branch alone
    block_off      zero the whole HB-MoE sublayer     -> is the block needed?
    none           control

The paper's appendix already answers a *different* question: it compares "with
MoE" against a parameter-matched dense baseline and finds the MoE wins with fewer
active parameters (Sum 4.012 vs 3.801).  That is the block_off contrast.  It does
not test whether the routing *decision* carries any of that gain, which is what
the first mode above isolates.

Only HB layers are touched.  AS layers are left alone: their gate reads
``data_mask``, which is constant for a single-embodiment benchmark, so they are a
constant router by construction (verified on 1682/1682 LIBERO control steps).
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
import torch.nn.functional as F

MODES = ("none", "router_random", "shared_off", "routed_off", "block_off")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--calvin-root", required=True)
    ap.add_argument("--wrist-layout", default="released-left")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from himoe_calvin_alignment.policy_server import build_parser, create_policy

    inner_args = build_parser().parse_args([
        "--checkpoint-dir", args.checkpoint_dir,
        "--upstream-root", args.upstream_root,
        "--calvin-root", args.calvin_root,
        "--wrist-layout", args.wrist_layout,
        "--host", args.host,
        "--port", str(args.port),
    ])
    validated = create_policy(inner_args)
    core = validated._policy.model
    expert_model = core.paligemma_with_expert.gemma_expert
    hb = [(i, layer.mlp) for i, layer in enumerate(expert_model.layers)
          if type(layer.mlp).__name__ == "HBMoE"]
    if not hb:
        raise SystemExit("no HBMoE layers found; the model structure changed")
    print("HB layers: %s | mode = %s" % ([i for i, _ in hb], args.mode), flush=True)

    stats = {"gate_calls": 0, "block_calls": 0}

    # ---- routing ablation: rewrite the gate's top-k choice -----------------
    if args.mode == "router_random":
        gen = torch.Generator(device="cuda").manual_seed(args.seed)

        def gate_hook(gate, inputs, output):
            idx, w, aux = output
            x = inputs[0]
            k = gate.top_k
            with torch.no_grad():
                logits = F.linear(x.reshape(-1, x.shape[-1]), gate.weight, None)
                probs = logits.softmax(-1)
                new_idx = torch.rand(probs.shape, generator=gen,
                                     device=probs.device).topk(k, dim=-1).indices
                p = probs.gather(-1, new_idx)
                new_w = p / (p.sum(-1, keepdim=True) + 1e-20)
            stats["gate_calls"] += 1
            return new_idx, new_w.to(w.dtype), aux

        for _, mlp in hb:
            mlp.gate.register_forward_hook(gate_hook)

    # ---- branch ablation: rewrite the block output -------------------------
    elif args.mode in ("shared_off", "routed_off", "block_off"):
        cache: dict[int, torch.Tensor] = {}

        def shared_hook(layer_idx):
            def hook(mod, inputs, output):
                cache[layer_idx] = output.detach()
                if args.mode in ("shared_off", "block_off"):
                    return torch.zeros_like(output)
                return None
            return hook

        def block_hook(layer_idx):
            def hook(mod, inputs, output):
                shared = cache.pop(layer_idx, None)
                if shared is None:
                    return None
                stats["block_calls"] += 1
                if args.mode == "block_off":
                    return torch.zeros_like(output)
                if args.mode == "routed_off":
                    # output currently holds routed + shared; keep shared only
                    return shared.to(output.dtype)
                return None  # shared_off: shared was already zeroed upstream
            return hook

        for i, mlp in hb:
            mlp.shared_experts.register_forward_hook(shared_hook(i))
            mlp.register_forward_hook(block_hook(i))

    validated._metadata["hb_ablation"] = args.mode
    validated._metadata["hb_ablation_seed"] = args.seed
    validated._metadata["hb_ablated_layers"] = [i for i, _ in hb]

    inner_infer = validated.infer

    def infer(observation):
        out = inner_infer(observation)
        n = stats["gate_calls"] + stats["block_calls"]
        if n and n % 20000 == 0:
            print("ablation fired %d times" % n, flush=True)
        return out

    validated.infer = infer

    from moevla.serving.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO, force=True)
    print("serving on ws://%s:%d" % (args.host, args.port), flush=True)
    WebsocketPolicyServer(policy=validated, host=args.host, port=args.port,
                          metadata=validated.metadata).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
