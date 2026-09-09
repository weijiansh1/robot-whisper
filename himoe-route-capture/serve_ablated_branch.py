"""Which branch of the HB-MoE block actually carries the behaviour?

HBMoE.forward is  y = moe_infer(routed) + shared_experts(identity)  so the two
branches can be removed independently with forward hooks, no source edit:

  --mode none            control
  --mode shared_off      shared_experts output -> 0   (routed branch only)
  --mode routed_off      routed output -> 0           (shared branch only)
  --mode shared_rescaled shared -> 0, routed scaled so the block output keeps
                         its original norm; separates "lost this function" from
                         "lost this much magnitude"
  --mode block_off       both branches -> 0; the whole HB MoE sublayer becomes a
                         no-op and only the residual stream passes through

Only HB layers have a shared expert (ASMoEConfig.n_shared_experts is None), so
AS blocks are untouched in every mode.
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="checkpoint-right")
    ap.add_argument("--mode", required=True,
                    choices=["none", "shared_off", "routed_off", "shared_rescaled",
                             "block_off"])
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root,
                           args.gpu, args.suite, args.libero_wrist_layout)
    expert_model = policy._policy.model.paligemma_with_expert.gemma_expert
    hb = [(i, l.mlp) for i, l in enumerate(expert_model.layers)
          if type(l.mlp).__name__ == "HBMoE"]
    print("HB layers:", [i for i, _ in hb], "| mode =", args.mode, flush=True)

    cache: dict[int, torch.Tensor] = {}
    stats = {"blocks": 0}

    def shared_hook(layer_idx):
        def hook(mod, inputs, output):
            if args.mode == "none":
                return None
            cache[layer_idx] = output.detach()
            if args.mode == "block_off":
                return torch.zeros_like(output)
            if args.mode in ("shared_off", "shared_rescaled"):
                return torch.zeros_like(output)
            return None

        return hook

    def block_hook(layer_idx):
        def hook(mod, inputs, output):
            if args.mode == "none":
                return None
            shared = cache.pop(layer_idx, None)
            if shared is None:
                return None
            stats["blocks"] += 1
            if args.mode == "block_off":
                # whole MoE sublayer becomes a no-op; only the residual survives
                return torch.zeros_like(output)
            if args.mode == "routed_off":
                # output currently holds routed + shared; keep shared only
                return shared.to(output.dtype)
            if args.mode == "shared_rescaled":
                # output currently holds routed only (shared was zeroed)
                with torch.no_grad():
                    target = (output.float() + shared.float()).norm()
                    cur = output.float().norm().clamp_min(1e-6)
                    return (output.float() * (target / cur)).to(output.dtype)
            return None

        return hook

    for i, mlp in hb:
        mlp.shared_experts.register_forward_hook(shared_hook(i))
        mlp.register_forward_hook(block_hook(i))

    policy.metadata["hb_branch_ablation"] = args.mode
    print("serving on ws://127.0.0.1:%d" % args.port, flush=True)
    PolicyServer(policy, "127.0.0.1", args.port, "himoe").serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
