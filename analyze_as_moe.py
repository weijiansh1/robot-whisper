#!/usr/bin/env python3
"""AS-MoE activation across rollouts of the same state, and what could change it.

The question is what the top level of the hierarchy does when the initial state
is held fixed and only the flow-noise draw varies.  The code answers it before
any data does: `ASMoE.forward` (modeling_moe.py:259-262) passes `data_mask` --
not the hidden state -- to its gate, and `data_mask` is a frozen dataclass field
read straight from the dataset config (libero_policy.py:59), so the AS routing
of a deployment is a constant of the deployment.  Everything in part 1 is
therefore a check that the capture agrees with that, not a search for structure.

  1. exhaustive: over every captured control step of all five tasks, is the AS
     router input, its choice and its probability bit-identical?  Checked at
     full resolution (4 layers x 10 denoise x 11 suffix), across scenes, across
     noise draws, across suites.
  2. what the four AS routers decide, and by what margin, in each of the four
     released LIBERO checkpoints.
  3. the decision table over the whole realistic input domain: config.py
     contains exactly four distinct data_masks, so the AS level's entire
     behaviour is sixteen numbers per checkpoint.
  4. the combine weight.  AS is top-1, and `norm_topk_prob` is guarded by
     `if self.top_k > 1` (modeling_moe.py:141), so the branch is scaled by the
     raw softmax probability rather than by 1.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import torch
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
RUN_ID = "right-16x32"
AS_LAYERS = [0, 1, 16, 17]
BF16_MANT = 8

#: the four distinct data_mask literals in the released training config, with
#: the datasets that carry them (training/config.py)
MASKS = {
    "eef7   (calvin_abc_eef, all libero_*, oxe)": [1] * 7 + [0] * 17,
    "calvin_joint (calvin_joint, calvin_d_joint)": [0] * 8 + [1] * 8 + [0] * 8,
    "aloha  (aloha/*)": [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1,
    "agibot (agibot_sim_joint)": [0] * 8 + [1] * 8 + [1] * 8,
}
CKPT = {"libero_goal": "HiMoE-VLA-Libero-Goal", "libero_spatial": "HiMoE-VLA-Libero-Spatial",
        "libero_object": "HiMoE-VLA-Libero-Object", "libero_10": "HiMoE-VLA-Libero-10"}


def bf16_ulp(x: float) -> float:
    """Spacing of bfloat16 at |x| -- the gate computes in bf16 (gate line 128)."""
    if x == 0:
        return 2.0 ** -133
    return 2.0 ** (np.floor(np.log2(abs(x))) - BF16_MANT)


def as_gates(name: str) -> dict[int, np.ndarray]:
    sd = torch.load(BRIDGE / "checkpoints" / name / "pytorch_model.pth",
                    map_location="cpu", weights_only=True, mmap=True)
    out = {}
    for k in sd:
        if k.endswith("mlp.gate.weight") and sd[k].shape == (3, 24):
            out[int(re.search(r"layers\.(\d+)\.", k).group(1))] = sd[k].float().numpy()
    if sorted(out) != AS_LAYERS:
        raise RuntimeError("expected AS gates at %s, found %s" % (AS_LAYERS, sorted(out)))
    return out


def decide(W: np.ndarray, mask) -> tuple[int, np.ndarray, float, float]:
    """Winner, softmax, top-1 minus top-2 logit gap, and that gap in bf16 ulp."""
    x = np.asarray(mask, np.float32)
    lg = x @ W.T
    p = np.exp(lg - lg.max())
    p /= p.sum()
    o = np.argsort(-lg)
    gap = float(lg[o[0]] - lg[o[1]])
    return int(o[0]), p, gap, gap / bf16_ulp(float(lg[o[0]]))


def part1() -> None:
    print("=== 1. is AS routing identical across every rollout? ===")
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    total_rows = total_eps = 0
    for suite in ("libero_goal", "libero_spatial", "libero_10", "libero_object"):
        base = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        if not base.exists():
            continue
        for task in sorted(base.iterdir()):
            run = task / RUN_ID
            if not (run / "server/routes.zarr").exists():
                continue
            z = zarr.open(str(run / "server/routes.zarr"), mode="r")
            ids = np.asarray(z["as_expert_ids"][:])
            prs = np.asarray(z["as_probs"][:])
            n = ids.shape[0]
            n_ep = len(json.loads((run / "client/summaries.json").read_text()))
            uniq_ids = np.unique(ids, axis=0)
            uniq_prs = np.unique(prs.reshape(n, -1), axis=0)

            # full resolution, straight from the router input
            h = zarr.open(str(run / "server/hidden.zarr"), mode="r")
            ah = h["as_hidden"]
            ref = np.asarray(ah[0])
            same = True
            for a in range(0, n, 512):
                blk = np.asarray(ah[a:min(a + 512, n)])
                if not np.array_equal(blk, np.broadcast_to(ref, blk.shape)):
                    same = False
                    break
            sm = json.loads((run / "server/capture_summary.json").read_text())
            print("  %-15s %-40s %5d eps %6d steps | expert tuples %d | prob "
                  "vectors %d | router input identical over 4x10x11: %s | "
                  "as_collapsed %s/%s"
                  % (suite, task.name[:40], n_ep, n, len(uniq_ids), len(uniq_prs),
                     same, sm.get("as_collapsed"), sm["control_steps"]))
            if len(uniq_ids) == 1:
                print("       tuple %s  probs %s"
                      % (uniq_ids[0].tolist(),
                         np.array2string(prs[0], precision=6)))
            total_rows += n
            total_eps += n_ep
    print("  total: %d episodes, %d control steps" % (total_eps, total_rows))


def part23() -> None:
    print("\n=== 2/3. what the four AS routers decide, per checkpoint ===")
    for suite, name in CKPT.items():
        if not (BRIDGE / "checkpoints" / name).exists():
            continue
        G = as_gates(name)
        print("\n%s" % name)
        for label, mask in MASKS.items():
            row = []
            for L in AS_LAYERS:
                e, p, gap, ulp = decide(G[L], mask)
                row.append("L%-2d->e%d (p=%.3f, gap %.1e = %.2f ulp)" % (L, e, p[e], gap, ulp))
            print("  %-44s %s" % (label, "  ".join(row[:2])))
            print("  %-44s %s" % ("", "  ".join(row[2:])))

    # do the four checkpoints share their AS gates?
    print("\n  are the AS gate weights the same across the four checkpoints?")
    have = [n for n in CKPT.values() if (BRIDGE / "checkpoints" / n).exists()]
    ref = as_gates(have[0])
    for name in have[1:]:
        G = as_gates(name)
        d = max(float(np.abs(G[L] - ref[L]).max()) for L in AS_LAYERS)
        print("    %-28s vs %-28s max|dW| = %.6f" % (name, have[0], d))


def part4() -> None:
    print("\n=== 4. the AS combine weight ===")
    G = as_gates(CKPT["libero_10"])
    tot = 1.0
    for L in AS_LAYERS:
        e, p, _, _ = decide(G[L], MASKS["eef7   (calvin_abc_eef, all libero_*, oxe)"])
        print("  layer %2d: expert %d, branch scaled by %.4f  (uniform floor 1/3 = 0.3333)"
              % (L, e, p[e]))
        tot *= p[e]
    print("  product over the four AS layers: %.4f" % tot)
    print("  AS has n_shared_experts=None, so nothing runs alongside to make it up.")


def main() -> int:
    part1()
    part23()
    part4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
