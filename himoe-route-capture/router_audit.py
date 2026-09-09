"""Router audit: is HB routing globally balanced but locally structured?

The four quantities that decide whether routing topology is worth studying at all,
each normalised so the "no information" value is unambiguous:

    H_load   = H(mean_u p_u) / log2(32)      1.0 => every expert equally used overall
    H_token  = mean_u H(p_u) / log2(32)      1.0 => each token's own softmax is flat
    M_4      = mean_u sum_{e in Top4} p_ue   0.125 => Top-4 absorbs no more than chance
    D_45     = mean_u (p_(4) - p_(5))        0    => the Top-4 boundary is a coin flip

The interesting regime is H_load ~ 1 with H_token < 1 and M_4 >> 0.125: balanced in
aggregate, conditional locally.  H_load ~ H_token ~ 1 with M_4 ~ 0.125 means the
router is degenerate and expert *identity* carries nothing, at which point the
thing to study is expert output contribution rather than expert ID.

M_4 and D_45 are computed from the stored full 32-dim softmax, not from the gate's
returned top-4: MoEGate renormalises the selected four to sum to 1 (modeling_moe.py,
`norm_topk_prob`), which destroys exactly the mass information M_4 measures.

Two precision caveats this reports rather than hides:

* the gate calls `torch.topk(..., sorted=False)`, so the stored expert-id array is
  NOT ordered by probability -- `ids[..., 0]` is an arbitrary member of the top-4.
  Anything that needs a genuine top-1 must recompute it from the probabilities.
* the probabilities are stored as float16, whose spacing near 1/32 is about
  3.05e-5.  If D_45 is at or below that, the top-4 boundary is not resolvable in
  the stored data and disagreements between the stored ids and an argsort of the
  stored probs are quantisation, not a capture bug.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

N_EXPERTS = 32
LOG2_E = float(np.log2(N_EXPERTS))
FLOAT16_SPACING_AT_1_32 = 2.0 ** -15  # 1/32 = 2^-5, 10 mantissa bits


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log2(p), axis=axis)


def audit(probs):
    """probs [n_sites, 32], each row a full softmax."""
    probs = probs / np.maximum(probs.sum(-1, keepdims=True), 1e-12)
    ordered = np.sort(probs, axis=-1)[:, ::-1]
    return {
        "H_load": float(entropy(probs.mean(0)) / LOG2_E),
        "H_token": float(np.mean(entropy(probs)) / LOG2_E),
        "M_4": float(np.mean(ordered[:, :4].sum(-1))),
        "D_45": float(np.mean(ordered[:, 3] - ordered[:, 4])),
        "D_45_over_float16_spacing": float(
            np.mean(ordered[:, 3] - ordered[:, 4]) / FLOAT16_SPACING_AT_1_32),
        "n_sites": int(probs.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default="analysis/router-audit/summary.json")
    args = ap.parse_args()

    report = {}
    for task in args.task:
        path = pathlib.Path(task)
        group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
        n_steps = min(args.max_steps, group["hb_router_probs"].shape[0])
        probs = np.asarray(group["hb_router_probs"][:n_steps], dtype=np.float32)
        ids = np.asarray(group["hb_expert_ids"][:n_steps])
        n_layers, n_suffix = probs.shape[1], probs.shape[3]

        print("\n=== %s ===" % path.name[:60])
        # how badly is the stored id array mis-ordered, and is that resolvable?
        gathered = np.take_along_axis(probs, ids.astype(np.int64), axis=-1)
        is_max = float(np.mean(gathered[..., 0] == gathered.max(-1)))
        stored_set = np.sort(ids.astype(np.int64), axis=-1)
        argsort_set = np.sort(np.argsort(-probs, axis=-1)[..., :4], axis=-1)
        agree = float(np.mean(np.all(stored_set == argsort_set, axis=-1)))
        print("  stored ids[...,0] is the true argmax in %.1f%% of sites "
              "(25%% = arbitrary member, 100%% = sorted)" % (100 * is_max))
        print("  stored top-4 set == argsort of stored probs in %.2f%% of sites" % (100 * agree))

        entry = {"ids_first_is_argmax": is_max, "set_agreement": agree, "layers": {}}
        print("  token   layer  H_load  H_token    M_4    D_45      D_45/fp16")
        for kind, token in (("action", slice(1, n_suffix)), ("state", slice(0, 1))):
            for layer in range(n_layers):
                stat = audit(probs[:, layer, :, token, :].reshape(-1, N_EXPERTS))
                entry["layers"]["%s_layer%d" % (kind, layer)] = stat
                print("  %-6s   %d     %.4f  %.4f  %.4f  %.2e   %6.1f"
                      % (kind, layer, stat["H_load"], stat["H_token"], stat["M_4"],
                         stat["D_45"], stat["D_45_over_float16_spacing"]), flush=True)
        report[path.name] = entry

    print("\nverdict rule: H_load~1 AND H_token<1 AND M_4>>0.125 -> routing topology "
          "is worth studying; H_token~1 AND M_4~0.125 -> study expert contribution instead")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
