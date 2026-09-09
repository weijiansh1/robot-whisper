#!/usr/bin/env python3
"""
Is the contrast suppression SELECTIVE?

A load-only gradient should attack the component of the gate that produces a
PERSISTENT expert preference, because that is what drives Cov(c,S) up at every
aggregation scale.  That component is the projection of W~ onto the mean router
input h_bar: it is identical for every token and every observation.

    E_h[Var_j z_j] = Var_j(w~_j . h_bar)      [fixed:       persistent bias]
                   + (1/N) Tr(W~ Cov(h) W~^T) [conditional: input-driven]

Prediction if HB-Reg shaped the gate: fixed suppressed >> conditional suppressed.
Internal control: the AS gates carry AS-Reg, a supervised CONTRASTIVE loss that
WANTS same-action-space tokens to route alike -- i.e. it wants a persistent
component.  It should show the opposite pattern.

Split by token type too: the state token (position 0) and the 10 action tokens have
very different router inputs, and a second-moment trace is dominated by whichever
is sharper.
"""
import math
from pathlib import Path

import numpy as np
import torch
import zarr

CKPT = Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
            "cache/checkpoints/HiMoE-VLA-Libero-Spatial/pytorch_model.pth")
RUN = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_spatial/"
           "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate/right-16x32")
HB, AS = [2, 3, 4, 5, 12, 13, 14, 15], [0, 1, 16, 17]


def split(W, hh, n_exp):
    """fixed / conditional decomposition of expert-logit contrast"""
    Wt = W - W.mean(0)
    hbar = hh.mean(0)
    dh = hh - hbar
    fixed = np.var(Wt @ hbar)
    cond = np.trace(Wt @ ((dh.T @ dh) / len(dh)) @ Wt.T) / n_exp
    return fixed, cond


def main():
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    W = {l: sd[[x for x in sd if f"layers.{l}.mlp.gate.weight" in x][0]]
         .float().numpy().astype(np.float64) for l in HB + AS}
    rng = np.random.default_rng(0)

    z = zarr.open(str(RUN / "server" / "hidden.zarr"), mode="r")
    idx = np.linspace(0, z["hb_hidden"].shape[0] - 1, 256).astype(int)
    Hhb = z["hb_hidden"].oindex[idx].astype(np.float32)     # (256,8,10,11,1024)
    Has = z["as_hidden"].oindex[idx].astype(np.float32)     # (256,4,10,11,24)

    for tag, toks in (("ALL 11 tokens", slice(None)),
                      ("state token only", slice(0, 1)),
                      ("10 action tokens", slice(1, 11))):
        print("=" * 104)
        print(f"HB gates (N=32, K=4) -- {tag}")
        print(f"  {'gate':<7} {'fixed':>10} {'cond':>10} {'cond frac':>10} | "
              f"{'kaiming fixed':>13} {'kaiming cond':>12} | "
              f"{'FIXED suppr':>12} {'COND suppr':>11} {'selectivity':>12}")
        for li, l in enumerate(HB):
            hh = Hhb[:, li, :, toks].reshape(-1, 1024).astype(np.float64)
            Wk = rng.uniform(-1 / 32, 1 / 32, size=(32, 1024))
            fx, cd = split(W[l], hh, 32)
            fk, ck = split(Wk, hh, 32)
            print(f"  L{l:<6} {fx:10.6f} {cd:10.6f} {cd/(fx+cd):10.4f} | "
                  f"{fk:13.6f} {ck:12.6f} | {fk/fx:11.1f}x {ck/cd:10.2f}x "
                  f"{(fk/fx)/(ck/cd):11.1f}x")
        print()

    print("=" * 104)
    print("AS gates (N=3, K=1) -- internal control, carries AS-Reg (contrastive), "
          "input dim 24")
    print(f"  {'gate':<7} {'fixed':>10} {'cond':>10} {'cond frac':>10} | "
          f"{'kaiming fixed':>13} {'kaiming cond':>12} | "
          f"{'FIXED suppr':>12} {'COND suppr':>11} {'selectivity':>12}")
    for li, l in enumerate(AS):
        hh = Has[:, li].reshape(-1, 24).astype(np.float64)
        b = 1 / math.sqrt(24)                       # kaiming_uniform_(a=sqrt(5)), d=24
        Wk = rng.uniform(-b, b, size=(3, 24))
        fx, cd = split(W[l], hh, 3)
        fk, ck = split(Wk, hh, 3)
        print(f"  L{l:<6} {fx:10.6f} {cd:10.6f} {cd/(fx+cd):10.4f} | "
              f"{fk:13.6f} {ck:12.6f} | {fk/fx:11.3f}x {ck/cd:10.3f}x "
              f"{(fk/fx)/(ck/cd):11.3f}x")
    print("\n  suppression = kaiming / actual;  selectivity = FIXED suppr / COND suppr.")
    print("  selectivity >> 1 means the persistent-bias component was attacked far")
    print("  harder than the input-driven one, which is the signature of a")
    print("  load-only teaching signal.  selectivity < 1 means the opposite.")


if __name__ == "__main__":
    main()
