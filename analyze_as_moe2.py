#!/usr/bin/env python3
"""Two things the first AS pass turned up that need settling in the right dtype.

A. A float32 recomputation of the Goal checkpoint puts layer 1's winner on
   expert 2, but the capture -- 1024 episodes of Goal -- records expert 0.  The
   gate casts to bfloat16 before the linear (`modeling_moe.py:128`), so the
   question is not which expert has the larger logit but which has the larger
   logit *after rounding*.  Redo the decision in bf16, the way the model does
   it, and report where the two dtypes disagree.  A disagreement is the finding:
   it means the choice is made by the rounding, not by the weights.

B. The four LIBERO checkpoints have different AS gate weights, yet all four
   route calvin_joint, aloha and agibot to exactly the same experts with
   identical probabilities.  The masks explain it if they have disjoint support:
   eef7 occupies dims 0-6 and the other three occupy 8-23, so a gradient from
   LIBERO data can only touch columns 0-6.  Check the weight difference column
   by column.
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import torch

BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
AS_LAYERS = [0, 1, 16, 17]
CKPT = {"Goal": "HiMoE-VLA-Libero-Goal", "Spatial": "HiMoE-VLA-Libero-Spatial",
        "Object": "HiMoE-VLA-Libero-Object", "Long": "HiMoE-VLA-Libero-10"}
MASKS = {
    "eef7": [1] * 7 + [0] * 17,
    "calvin_joint": [0] * 8 + [1] * 8 + [0] * 8,
    "aloha": [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1,
    "agibot": [0] * 8 + [1] * 8 + [1] * 8,
}


def as_gates(name):
    sd = torch.load(BRIDGE / "checkpoints" / name / "pytorch_model.pth",
                    map_location="cpu", weights_only=True, mmap=True)
    return {int(re.search(r"layers\.(\d+)\.", k).group(1)): sd[k].clone()
            for k in sd if k.endswith("mlp.gate.weight") and sd[k].shape == (3, 24)}


def main() -> int:
    print("=== A. the same decision in bf16 (what the model does) and f32 ===")
    print("captured tuple, every suite, 2560 episodes: [2, 0, 0, 1]\n")
    print("ckpt      layer  bf16 logits                       bf16 pick  "
          "f32 pick  agree")
    for tag, name in CKPT.items():
        G = as_gates(name)
        for L in AS_LAYERS:
            W = G[L]
            x = torch.tensor(MASKS["eef7"], dtype=torch.float32)
            # the model's path: cast the router input to bf16, then F.linear
            lg_b = torch.nn.functional.linear(x.to(torch.bfloat16),
                                              W.to(torch.bfloat16), None)
            # torch.topk on the softmax; softmax is monotone so the logits decide
            pick_b = int(torch.topk(lg_b.float().softmax(-1), 1).indices[0])
            lg_f = (x @ W.float().T)
            pick_f = int(lg_f.argmax())
            mark = "yes" if pick_b == pick_f else "NO  <-- decided by rounding"
            print("%-9s L%-4d  %-32s e%d         e%d        %s"
                  % (tag, L, np.array2string(lg_b.float().numpy(), precision=5),
                     pick_b, pick_f, mark))
        print()

    print("=== B. where do the four checkpoints differ? ===")
    ref = as_gates(CKPT["Goal"])
    supp = {k: sorted(i for i, v in enumerate(m) if v) for k, m in MASKS.items()}
    print("mask supports: eef7 %s | others %s (dim 7 unused by every mask)"
          % (supp["eef7"], sorted(set(supp["calvin_joint"]) | set(supp["aloha"])
                                  | set(supp["agibot"]))))
    for tag, name in list(CKPT.items())[1:]:
        G = as_gates(name)
        # the gate weights are stored in bf16; numpy has no bf16, so lift to f32
        d = torch.stack([(G[L].float() - ref[L].float()).abs()
                         for L in AS_LAYERS]).amax((0, 1))
        cols = np.flatnonzero(d.numpy() > 0)
        print("  %-8s vs Goal: columns that moved = %s   max|dW| there %.6f, "
              "elsewhere %.1e"
              % (tag, cols.tolist(), d[cols].max() if len(cols) else 0.0,
                 d[[c for c in range(24) if c not in cols]].max() if len(cols) < 24 else 0.0))

    print("\n=== C. the whole AS decision table (bf16, as deployed) ===")
    print("mask           " + "  ".join("%-8s" % t for t in CKPT))
    for mk, mask in MASKS.items():
        row = []
        for tag, name in CKPT.items():
            G = as_gates(name)
            t = []
            for L in AS_LAYERS:
                lg = torch.nn.functional.linear(
                    torch.tensor(mask, dtype=torch.bfloat16),
                    G[L].to(torch.bfloat16), None)
                t.append(int(torch.topk(lg.float().softmax(-1), 1).indices[0]))
            row.append("".join(str(v) for v in t))
        print("  %-13s" % mk + "  ".join("%-8s" % r for r in row))
    print("  (each cell is the expert chosen at layers 0, 1, 16, 17)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
