"""How different are the 32 HB-MoE experts from each other, in the weights?

If the experts are mostly a shared component plus small per-expert deltas, then
a flat router is not "well balanced" -- it is close to irrelevant, because every
top-4 choice computes nearly the same function.

Decomposition per layer:  W_e = W_mean + D_e
  ratio = ||D_e|| / ||W_mean||   -- how much of an expert is its own
Run in the model env (torch).
"""

from __future__ import annotations

import json

import numpy as np
import torch

CKPT = ("/home/jovyan/.cache/himoe-libero-bridge/checkpoints/"
        "HiMoE-VLA-Libero-Goal/pytorch_model.pth")
HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
PROJ = ["gate_proj", "up_proj", "down_proj"]


def main():
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
    prefix = "paligemma_with_expert.gemma_expert.layers"
    out = {}

    print("%-6s %-10s %-12s %-12s %-12s %s" %
          ("layer", "proj", "||D||/||M||", "cos 均值", "cos 范围", "shared/routed 范数比"))
    for L in HB_LAYERS:
        rec = {}
        for proj in PROJ:
            W = torch.stack([
                sd["%s.%d.mlp.experts.%d.%s.weight" % (prefix, L, e, proj)].float()
                for e in range(32)
            ])                                   # [32, out, in]
            flat = W.reshape(32, -1)
            mean = flat.mean(0)
            delta = flat - mean
            ratio = float((delta.norm(dim=1) / mean.norm()).mean())

            # pairwise cosine between full expert weight vectors
            n = flat / flat.norm(dim=1, keepdim=True)
            cos = (n @ n.T)
            iu = torch.triu_indices(32, 32, offset=1)
            cvals = cos[iu[0], iu[1]]

            shared = sd["%s.%d.mlp.shared_experts.%s.weight" % (prefix, L, proj)].float()
            share_ratio = float(shared.norm() / flat.norm(dim=1).mean())

            rec[proj] = {
                "delta_over_mean": ratio,
                "cos_mean": float(cvals.mean()),
                "cos_min": float(cvals.min()),
                "cos_max": float(cvals.max()),
                "shared_over_routed_norm": share_ratio,
            }
            print("%-6d %-10s %-12.4f %-12.4f [%.3f, %.3f]  %.3f"
                  % (L, proj, ratio, cvals.mean(), cvals.min(), cvals.max(), share_ratio))
            del W, flat, n, cos
        out[L] = rec
        print()

    # router gate: fixed preference vs input-driven
    print("=== HB router gate 权重结构 ===")
    print("%-6s %-14s %-14s %s" % ("layer", "行范数均值", "行范数变异", "行间余弦均值"))
    for L in HB_LAYERS:
        g = sd["%s.%d.mlp.gate.weight" % (prefix, L)].float()   # [32, 1024]
        rn = g.norm(dim=1)
        n = g / rn[:, None]
        cos = n @ n.T
        iu = torch.triu_indices(32, 32, offset=1)
        print("%-6d %-14.4f %-14.4f %.4f"
              % (L, rn.mean(), rn.std() / rn.mean(), cos[iu[0], iu[1]].mean()))
        out[L]["gate"] = {"row_norm_mean": float(rn.mean()),
                          "row_norm_cv": float(rn.std() / rn.mean()),
                          "row_cos_mean": float(cos[iu[0], iu[1]].mean())}

    json.dump(out, open("/home/jovyan/work/himoe-route-capture/expert_weights.json", "w"),
              indent=2)
    print("\n参考：两个独立随机初始化的高维向量，余弦应当接近 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
