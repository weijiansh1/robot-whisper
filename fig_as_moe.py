#!/usr/bin/env python3
"""One figure for AS-MoE: what it does, what it cannot do, and where it ties."""

from __future__ import annotations

import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FIG = pathlib.Path(__file__).resolve().parent / "fig"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
AS_LAYERS = [0, 1, 16, 17]
CKPT = {"Goal": "HiMoE-VLA-Libero-Goal", "Spatial": "HiMoE-VLA-Libero-Spatial",
        "Object": "HiMoE-VLA-Libero-Object", "Long": "HiMoE-VLA-Libero-10"}
MASKS = {"eef7\n(libero, calvin-eef, oxe)": [1] * 7 + [0] * 17,
         "calvin_joint": [0] * 8 + [1] * 8 + [0] * 8,
         "aloha": [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1,
         "agibot": [0] * 8 + [1] * 8 + [1] * 8}


def as_gates(name):
    sd = torch.load(BRIDGE / "checkpoints" / name / "pytorch_model.pth",
                    map_location="cpu", weights_only=True, mmap=True)
    return {int(re.search(r"layers\.(\d+)\.", k).group(1)): sd[k].clone()
            for k in sd if k.endswith("mlp.gate.weight") and sd[k].shape == (3, 24)}


def bf16_logits(W, mask):
    return torch.nn.functional.linear(torch.tensor(mask, dtype=torch.bfloat16),
                                      W.to(torch.bfloat16), None).float().numpy()


def main() -> int:
    FIG.mkdir(exist_ok=True)
    G = {t: as_gates(n) for t, n in CKPT.items()}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    # A: softmax over the 3 experts, LIBERO's own mask, four checkpoints
    a = ax[0]
    w = .2
    for j, t in enumerate(CKPT):
        for i, L in enumerate(AS_LAYERS):
            lg = bf16_logits(G[t][L], MASKS["eef7\n(libero, calvin-eef, oxe)"])
            p = np.exp(lg - lg.max())
            p /= p.sum()
            for e in range(3):
                a.bar(i + (j - 1.5) * w, p[e], w * .9,
                      bottom=p[:e].sum(), color=plt.cm.tab10(e),
                      edgecolor="w", lw=.4)
    a.axhline(1 / 3, ls="--", c="k", lw=.9)
    a.axhline(2 / 3, ls="--", c="k", lw=.9)
    a.set_xticks(range(4), ["L%d" % L for L in AS_LAYERS])
    a.set_ylabel("router probability")
    a.set_title("A  AS routers on LIBERO's own mask\n"
                "4 bars = Goal/Spatial/Object/Long; dashes = uniform thirds",
                fontsize=10)
    a.text(1, 1.03, "layer 1 is flat in all four", ha="center", fontsize=8)

    # B: the winning margin, in units of the arithmetic's own resolution
    a = ax[1]
    for mi, (mk, mask) in enumerate(MASKS.items()):
        gaps = []
        for L in AS_LAYERS:
            lg = bf16_logits(G["Long"][L], mask)
            s = np.sort(lg)[::-1]
            gaps.append(max(s[0] - s[1], 1e-9))
        a.plot(range(4), gaps, "o-", label=mk.split("\n")[0], ms=6)
    a.set_yscale("log")
    a.axhspan(1e-9, 2.0 ** -11, color="tab:red", alpha=.15)
    a.text(1.5, 1.5e-9, "below one bf16 ulp: the expert is\n"
                        "picked by the tie-break, not the weights",
           fontsize=8, color="tab:red")
    a.set_xticks(range(4), ["L%d" % L for L in AS_LAYERS])
    a.set_ylabel("top-1 minus top-2 logit (bf16)")
    a.legend(fontsize=8); a.set_title("B  how decided is each choice?  (Long ckpt)",
                                      fontsize=10)

    # C: the entire decision table
    a = ax[2]
    sigs, grid = {}, np.zeros((len(MASKS), len(CKPT)))
    txt = []
    for mi, (mk, mask) in enumerate(MASKS.items()):
        row = []
        for cj, t in enumerate(CKPT):
            s = "".join(str(int(np.argmax(bf16_logits(G[t][L], mask))))
                        for L in AS_LAYERS)
            sigs.setdefault(s, len(sigs))
            grid[mi, cj] = sigs[s]
            row.append(s)
        txt.append(row)
    a.imshow(grid, cmap="Set2", vmin=0, vmax=7)
    for i in range(len(MASKS)):
        for j in range(len(CKPT)):
            a.text(j, i, txt[i][j], ha="center", va="center", fontsize=11)
    a.set_xticks(range(4), list(CKPT))
    a.set_yticks(range(4), [m.split("\n")[0] for m in MASKS])
    a.set_title("C  every AS decision the released model can make\n"
                "(expert at layers 0,1,16,17)  -- two signatures, not four",
                fontsize=10)

    fig.suptitle("AS-MoE routes on data_mask alone, so it is constant within a "
                 "deployment: 2560 rollouts, 51308 control steps, one tuple [2,0,0,1]",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "as1_decision.png", dpi=130)
    print("wrote", FIG / "as1_decision.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
