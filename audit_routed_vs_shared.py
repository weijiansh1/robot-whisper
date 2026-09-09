#!/usr/bin/env python3
"""H1 vs H2 for "the routed branch contributes little".

H1 mechanistic redundancy: routed(h) is (near) a linear re-dress of shared(h)
   -- ablating either branch removes a duplicated function, hence no effect.
H2 downstream compensation: routed(h) is a genuinely different function; the
   null ablation effect must then come from closed-loop robustness.

On real router inputs (hidden.zarr) with the deployment's authoritative top-4
(routes.zarr ids + pre-norm weights, renormalized), per HB layer and token
type: norm ratio, cos(routed, shared), routed share of block output, and
held-out R^2 of ridge maps  shared->routed  and  h->routed  (PCA-128).
Goal checkpoint / goal_mid run.  Writes audit_routed_vs_shared.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import zarr

HERE = pathlib.Path(__file__).resolve().parent
RUN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    HERE / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_goal"
    / "open_the_middle_drawer_of_the_cabinet/right-16x32/server")
CKPT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
    "/home/jovyan/.cache/himoe-libero-bridge/checkpoints"
    "/HiMoE-VLA-Libero-Goal/pytorch_model.pth")
TAG = sys.argv[3] if len(sys.argv) > 3 else "goal_mid"
PRE = "paligemma_with_expert.gemma_expert.layers"
HB = [2, 3, 4, 5, 12, 13, 14, 15]
N_ROWS = 1200
RNG = np.random.default_rng(0)


def silu(x):
    return x / (1.0 + np.exp(-x))


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    P = alpha * np.eye(A.shape[1]); P[0, 0] = 0
    W = np.linalg.solve(A.T @ A + P, A.T @ Ytr)
    pred = np.c_[np.ones(len(Xte)), (Xte - mu) / sd] @ W
    return 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()


def pca_fit(X, k=128):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def main() -> int:
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
    keys = list(sd.keys())
    shared_key = [k for k in keys if "2.mlp" in k and "shared" in k][:3]
    gate_key = [k for k in keys if "layers.2.mlp" in k and "gate.weight" in k]
    print("shared naming sample:", shared_key)
    print("router gate sample:", gate_key)

    zg = zarr.open_group(str(RUN.parent / "server/routes.zarr")
                         if not (RUN / "routes.zarr").exists()
                         else str(RUN / "routes.zarr"), mode="r")
    zh = zarr.open_group(str(RUN / "hidden.zarr"), mode="r")
    N = zh["hb_hidden"].shape[0]
    rows = np.sort(RNG.choice(N, N_ROWS, replace=False))
    # sites: (denoise, token) pairs -- state token once (bit-constant over
    # denoise), one mid action token at early and late denoise
    SITES = [(0, 0, "state_d0"), (0, 5, "action_d0"), (9, 5, "action_d9")]
    H = np.asarray(zh["hb_hidden"].oindex[rows, :, [0, 9], :, :], np.float32)
    IDS = np.asarray(zg["hb_expert_ids"].oindex[rows, :, [0, 9], :, :], np.int64)
    WSEL = np.asarray(zg["hb_selected_prob"].oindex[rows, :, [0, 9], :, :],
                      np.float32)
    d_index = {0: 0, 9: 1}

    out = {}
    for li, L in enumerate(HB):
        base = "%s.%d.mlp" % (PRE, L)
        Wg = {e: sd["%s.experts.%d.gate_proj.weight" % (base, e)].float().numpy()
              for e in range(32)}
        Wu = {e: sd["%s.experts.%d.up_proj.weight" % (base, e)].float().numpy()
              for e in range(32)}
        Wd = {e: sd["%s.experts.%d.down_proj.weight" % (base, e)].float().numpy()
              for e in range(32)}
        Sg = sd["%s.shared_experts.gate_proj.weight" % base].float().numpy()
        Su = sd["%s.shared_experts.up_proj.weight" % base].float().numpy()
        Sdn = sd["%s.shared_experts.down_proj.weight" % base].float().numpy()

        out[L] = {}
        for d, tok, label in SITES:
            h = H[:, li, d_index[d], tok]                      # (n,1024)
            ids = IDS[:, li, d_index[d], tok]                  # (n,4)
            w = WSEL[:, li, d_index[d], tok]
            w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
            shared = silu(h @ Sg.T) * (h @ Su.T) @ Sdn.T
            routed = np.zeros_like(shared)
            for e in range(32):
                m, slot = np.where(ids == e)
                if not len(m):
                    continue
                he = h[m]
                oe = silu(he @ Wg[e].T) * (he @ Wu[e].T) @ Wd[e].T
                routed[m] += w[m, slot][:, None] * oe
            nr = np.linalg.norm(routed, axis=1)
            ns = np.linalg.norm(shared, axis=1)
            cos = (routed * shared).sum(1) / np.maximum(nr * ns, 1e-9)
            blk = routed + shared
            share = nr / np.maximum(np.linalg.norm(blk, axis=1), 1e-9)
            cos_keep = (blk * shared).sum(1) / np.maximum(
                np.linalg.norm(blk, axis=1) * ns, 1e-9)
            tr = np.arange(len(h)) % 2 == 0
            te = ~tr
            mu_s, V_s = pca_fit(shared[tr]); mu_r, V_r = pca_fit(routed[tr])
            mu_h, V_h = pca_fit(h[tr])
            Str, Ste = (shared[tr] - mu_s) @ V_s.T, (shared[te] - mu_s) @ V_s.T
            Rtr, Rte = (routed[tr] - mu_r) @ V_r.T, (routed[te] - mu_r) @ V_r.T
            Htr, Hte = (h[tr] - mu_h) @ V_h.T, (h[te] - mu_h) @ V_h.T
            r2_s = ridge_r2(Str, Rtr, Ste, Rte)
            r2_h = ridge_r2(Htr, Rtr, Hte, Rte)
            out[L][label] = {
                "norm_ratio_med": float(np.median(nr / np.maximum(ns, 1e-9))),
                "cos_med": float(np.median(cos)),
                "routed_share_of_block_med": float(np.median(share)),
                "cos_block_vs_shared_med": float(np.median(cos_keep)),
                "R2_shared_to_routed": float(r2_s),
                "R2_h_to_routed": float(r2_h),
            }
        r = out[L]["action_d0"]
        print("L%-3d action_d0: |r|/|s| %.2f  cos %.2f  share %.2f  "
              "cos(blk,s) %.3f  R2 s->r %.3f  R2 h->r %.3f"
              % (L, r["norm_ratio_med"], r["cos_med"],
                 r["routed_share_of_block_med"], r["cos_block_vs_shared_med"],
                 r["R2_shared_to_routed"], r["R2_h_to_routed"]))

    dst = HERE / ("audit_routed_vs_shared_%s.json" % TAG)
    dst.write_text(json.dumps(out, indent=1))
    print("\nwrote", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
